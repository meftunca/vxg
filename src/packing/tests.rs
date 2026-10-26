    use super::*;

    fn command_with(field: usize, value: u32) -> MetadataCommand {
        let mut values = [0u32; FIELD_COUNT];
        values[0] = 7;
        values[field] = value;
        command(values).unwrap()
    }

    fn roundtrip(layout: Layout, commands: &[MetadataCommand]) {
        let packed = pack_blocks(&[commands.to_vec()], layout, 32).unwrap();
        assert_eq!(unpack_blocks(&packed, layout).unwrap(), [commands.to_vec()]);
    }

    #[test]
    fn every_required_width_roundtrips_fixed_and_bitsliced() {
        for bits in [0u8, 1, 4, 8, 16, 32] {
            let high = if bits == 0 {
                0
            } else if bits == 32 {
                u32::MAX
            } else {
                (1 << bits) - 1
            };
            let commands = [command_with(3, 0), command_with(3, high)];
            roundtrip(Layout::Fixed, &commands);
            roundtrip(Layout::Bitsliced, &commands);
        }
    }

    #[test]
    fn int4_escapes_and_odd_tail_are_exact() {
        let commands = [command_with(1, 0), command_with(1, 14), command_with(1, 15)];
        roundtrip(Layout::Int4, &commands);
        let mut bytes = pack_int4_group(&commands).unwrap();
        let nibble_start = 1 + FIELD_COUNT * 4;
        bytes[nibble_start + 1] |= 0x80;
        assert!(unpack_int4_group(&mut Cursor::new(&bytes)).is_err());
    }

    #[test]
    fn ternary_accepts_every_opcode_and_rejects_reserved_selector() {
        let commands: Vec<_> = (0..=7)
            .map(|op| MetadataCommand {
                op,
                lit: op as u32,
                run: 8 + op as u32,
                distance: 17 + op as u32,
                seed_lo: op as u32,
                seed_hi: 0,
            })
            .collect();
        roundtrip(Layout::Ternary2Bit, &commands);
        roundtrip(Layout::TernaryBase3, &commands);
        let mut bytes = pack_ternary_group(&commands[..5], Selector::Base3).unwrap();
        let selector_offset = 1 + NUMERIC_FIELD_COUNT * 4;
        bytes[selector_offset] = 243;
        assert!(unpack_ternary_group(&mut Cursor::new(&bytes), Selector::Base3).is_err());
    }

    #[test]
    fn bitslice_is_byte_identical_to_fixed_under_the_same_plan() {
        let commands: Vec<_> = (0..32)
            .map(|index| MetadataCommand {
                op: if index & 1 == 0 { 7 } else { 2 },
                lit: (index % 5) as u32,
                run: 8 + (index % 17) as u32,
                distance: 31 + (index * 37) as u32,
                seed_lo: 0,
                seed_hi: 0,
            })
            .collect();
        assert_eq!(
            pack_fixed_group(&commands, false).unwrap().len(),
            pack_fixed_group(&commands, true).unwrap().len()
        );
    }

    #[test]
    fn evaluation_roundtrips_an_actual_encoder_trace() {
        let data = (0..16_384)
            .flat_map(|index| {
                format!("{{\"event\":\"update\",\"id\":{}}}\n", index % 47).into_bytes()
            })
            .collect::<Vec<_>>();
        let encoded = crate::compress(
            &data,
            &crate::Options {
                block_size: 4096,
                checksum: false,
                legacy_v2: true,
                ..crate::Options::default()
            },
        )
        .unwrap();
        let frame = Frame::parse(&encoded, data.len()).unwrap();
        let evaluation = evaluate_frame(&frame).unwrap();
        assert_eq!(evaluation.frame_bytes, encoded.len());
        assert_eq!(
            evaluation.results.len(),
            Layout::ALL.len() * GROUP_SIZES.len()
        );
        assert!(evaluation.results.iter().all(|result| result.commands > 0));
        assert!(!evaluation.distance_results.is_empty());
    }

    #[test]
    fn distance_models_roundtrip_and_reject_noncanonical_uleb() {
        let commands = vec![
            MetadataCommand { op: 7, lit: 0, run: 8, distance: 31, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 2, lit: 1, run: 9, distance: 31, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 7, lit: 0, run: 16, distance: 130_000, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 0, lit: 4, run: 0, distance: 0, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 1, lit: 0, run: 12, distance: 65_535, seed_lo: 0, seed_hi: 0 },
        ];
        let blocks = vec![commands];
        for model in DistanceModel::ALL {
            let packed = pack_distance_blocks(&blocks, model, 32).unwrap();
            assert_eq!(
                unpack_distance_blocks(&packed, model).unwrap(),
                expected_distances(&blocks).unwrap()
            );
        }
        for group_size in GROUP_SIZES {
            let packed = pack_distance_blocks(&blocks, DistanceModel::Dictionary16, group_size).unwrap();
            assert_eq!(
                unpack_distance_blocks(&packed, DistanceModel::Dictionary16).unwrap(),
                expected_distances(&blocks).unwrap()
            );
        }
        assert!(read_uleb(&mut Cursor::new(&[0x80, 0])).is_err());
    }

    #[test]
    fn bounded_entropy_roundtrips_mixed_commands_and_rejects_table_reordering() {
        let commands = vec![
            MetadataCommand { op: 0, lit: 7, run: 0, distance: 0, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 7, lit: 0, run: 8, distance: 31, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 2, lit: 1, run: 9, distance: 31, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 3, lit: 0, run: 17, distance: 0, seed_lo: 0x0000_4241, seed_hi: 0 },
            MetadataCommand { op: 1, lit: 2, run: 33, distance: 130_000, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 6, lit: 0, run: 64, distance: 0, seed_lo: 0x0403_0201, seed_hi: 0x0807_0605 },
        ];
        let packed = pack_entropy_blocks(&[commands.clone()]).unwrap();
        assert_eq!(unpack_entropy_blocks(&packed).unwrap(), [commands]);
        let mut malformed = pack_entropy_block(&[MetadataCommand { op: 7, lit: 0, run: 8, distance: 1, seed_lo: 0, seed_hi: 0 }, MetadataCommand { op: 2, lit: 0, run: 8, distance: 1, seed_lo: 0, seed_hi: 0 }]).unwrap();
        // Command count (4 bytes), then the opcode table length (2 bytes). The
        // first table is canonical length/symbol order; swap its symbols only.
        malformed.swap(6, 11);
        assert!(unpack_entropy_block(&malformed).is_err());
    }

    #[test]
    fn adaptive_entropy_charges_and_roundtrips_both_block_modes() {
        let tiny = vec![
            MetadataCommand { op: 7, lit: 0, run: 8, distance: 17, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 2, lit: 0, run: 8, distance: 17, seed_lo: 0, seed_hi: 0 },
        ];
        let repeated = (0..2048)
            .map(|index| {
                let mut command = MetadataCommand {
                    op: 7,
                    lit: 0,
                    run: 16,
                    distance: 31,
                    seed_lo: 0,
                    seed_hi: 0,
                };
                // A few high residuals force fixed-width planes wide, while
                // the remaining commands keep every field highly skewed.
                // Repeat the same skewed distribution in each 128-command
                // fixed group. Fixed-width packing must pay the outlier width
                // for all commands in every group; the entropy codebooks pay
                // their tables once for the whole LZ block.
                match index % 128 {
                    0..=7 => command.op = 1,
                    8..=15 => command.lit = 1 << (index - 1),
                    16..=23 => command.run = 1 << (index - 9),
                    24..=31 => command.distance = 1 << (index - 17),
                    32..=39 => command.seed_lo = 1 << (index - 25),
                    40..=47 => command.seed_hi = 1 << (index - 33),
                    _ => {}
                }
                command
            })
            .collect::<Vec<_>>();
        let blocks = vec![tiny, repeated];
        let packed = pack_adaptive_entropy_blocks(&blocks).unwrap();
        assert_eq!(unpack_adaptive_entropy_blocks(&packed).unwrap(), blocks);
        assert!(packed.fixed_blocks > 0);
        assert!(packed.entropy_blocks > 0);
        assert!(packed.max_block_scratch_bytes > packed.metadata_bytes / 2);
    }

    #[test]
    fn literal_entropy_roundtrips_all_literal_modes_and_rejects_malformed_tables() {
        let tokens = (0..16)
            .map(|token| {
                std::array::from_fn(|index| (token * TOKEN_WIDTH + index) as u8)
            })
            .collect::<Vec<[u8; TOKEN_WIDTH]>>();
        let dictionary_friendly = (0..32)
            .flat_map(|_| tokens.iter().flat_map(|token| token.iter().copied()))
            .collect::<Vec<u8>>();
        let mut state = 0x6a09e667f3bcc909u64;
        let entropy_friendly = (0..512)
            .map(|_| {
                state = state
                    .wrapping_mul(0x9e3779b97f4a7c15)
                    .wrapping_add(0x94d049bb133111eb);
                b"abcd"[(state >> 62) as usize]
            })
            .collect::<Vec<u8>>();
        let blocks = vec![
            Vec::new(),
            entropy_friendly,
            vec![b'a'; 512],
            dictionary_friendly,
            (0..64).collect::<Vec<u8>>(),
        ];
        let packed = pack_literal_entropy_blocks(&blocks).unwrap();
        assert_eq!(unpack_literal_entropy_blocks(&packed).unwrap(), blocks);
        assert!(packed.raw_blocks > 0);
        assert!(packed.entropy_blocks > 0);
        assert!(packed.dictionary_blocks > 0);

        let entropy_index = packed
            .blocks
            .iter()
            .position(|block| block.first() == Some(&1))
            .unwrap();
        let mut malformed = packed;
        malformed.blocks[entropy_index][0] = 2;
        assert!(unpack_literal_entropy_blocks(&malformed).is_err());

        let mut malformed = pack_literal_entropy_blocks(&blocks).unwrap();
        let dictionary_index = malformed
            .blocks
            .iter()
            .position(|block| block.first() == Some(&2))
            .unwrap();
        let first_token = malformed.blocks[dictionary_index][2..2 + TOKEN_WIDTH].to_vec();
        malformed.blocks[dictionary_index][2 + TOKEN_WIDTH..2 + TOKEN_WIDTH * 2]
            .copy_from_slice(&first_token);
        assert!(unpack_literal_entropy_blocks(&malformed).is_err());
    }

