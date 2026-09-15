use vxg::{compress,compress_with_stats,decompress,DecodeOptions,Options,Frame,Simd,max_compressed_size};
fn rng(s:&mut u64)->u8{*s^=*s<<13;*s^=*s>>7;*s^=*s<<17;*s as u8}
fn fixture(n:usize,period:usize)->Vec<u8>{
    let mut seed=0x123456789abcdef;
    let pat:Vec<u8>=(0..period.max(1)).map(|_|rng(&mut seed)).collect();
    (0..n).map(|i|if period==0{rng(&mut seed)}else{pat[i%period]}).collect()
}
#[test]
fn crc32c_known_answer_and_hardware_equivalence(){
    assert_eq!(vxg::checksum::software(b"123456789"),0xe3069283);
    assert_eq!(vxg::checksum::crc32c(b"123456789"),0xe3069283);
    assert_eq!(vxg::checksum::crc32c(b""),0);
    for n in 0..257{let data=fixture(n,0);assert_eq!(vxg::checksum::crc32c(&data),vxg::checksum::software(&data));}
}
#[test]
fn roundtrip_bound_and_linear_work_properties(){
    for n in [0,1,7,8,15,31,32,127,255,256,1023,4095,4096,4097,16393]{
        for p in [0,1,2,3,4,7,8,17,63,65,127]{for acceleration in [1,4,16,64]{
            let data=fixture(n,p);let opt=Options{block_size:4096,acceleration,..Options::default()};
            let (encoded,s)=compress_with_stats(&data,&opt).unwrap();
            assert!(encoded.len()<=max_compressed_size(data.len(),opt.block_size).unwrap());
            assert!(s.probes<=n as u64,"one probe per monotonically advanced position");
            assert!(s.matched_bytes<=n as u64,"accepted match regions are disjoint");
            assert!(s.accepted_matches<=(n/8) as u64);
            assert_eq!(decompress(&encoded,&DecodeOptions::default()).unwrap(),data);
            Frame::parse(&encoded,n).unwrap().validate_commands().unwrap();
        }}
    }
}
#[test]
fn exhaustive_binary_small_inputs(){
    // This is finite-domain testing, not a proof of the unbounded whole codec.
    // Inputs shorter than the metadata overhead exercise the RAW boundary.
    for n in 0..=10{for word in 0..(1usize<<n){
        let data:Vec<u8>=(0..n).map(|i|((word>>i)&1) as u8).collect();
        let frame=compress(&data,&Options::default()).unwrap();
        assert_eq!(decompress(&frame,&DecodeOptions::default()).unwrap(),data);
    }}
}
#[test]
fn deterministic_across_threads_and_cpu_dispatch(){
    let data=fixture(180_013,37);
    let reference=compress(&data,&Options{block_size:4096,simd:Simd::Scalar,..Options::default()}).unwrap();
    for threads in [1,2,4]{for simd in [Simd::Scalar,Simd::Sse2,Simd::Avx2,Simd::Avx512,Simd::Neon]{
        if !vxg::simd::Engine::supported(simd){continue;}
        let f=compress(&data,&Options{block_size:4096,threads,simd,..Options::default()}).unwrap();
        assert_eq!(f,reference);
        assert_eq!(decompress(&f,&DecodeOptions{threads,simd,..DecodeOptions::default()}).unwrap(),data);
    }}
}
#[test]
fn simd_prefix_exactness_and_unaligned_tails(){
    for simd in [Simd::Scalar,Simd::Sse2,Simd::Avx2,Simd::Avx512,Simd::Neon]{
        if !vxg::simd::Engine::supported(simd){continue;}
        let e=vxg::simd::Engine::new(simd).unwrap();
        for offset in 0..64{for n in 0..257{
            let a=fixture(n+offset+1,0);let mut b=a.clone();
            assert_eq!(e.lcp(&a[offset..offset+n],&b[offset..offset+n]),n);
            if n>0{let j=n/2;b[offset+j]^=1;assert_eq!(e.lcp(&a[offset..offset+n],&b[offset..offset+n]),j);}
            let mut dst=vec![0xab;n+32];e.copy(&mut dst[16..16+n],&a[offset..offset+n]);
            assert!(dst[..16].iter().chain(dst[16+n..].iter()).all(|&b|b==0xab));
            assert_eq!(&dst[16..16+n],&a[offset..offset+n]);
        }}
    }
}
#[test]
fn periodic_copy_preserves_non_power_of_two_phase(){
    let e=vxg::simd::Engine::new(Simd::Auto).unwrap();
    for d in 1..=64{let seed:Vec<u8>=(0..d).map(|x|x as u8).collect();
        for n in [0,1,31,63,64,65,127,256,257,1025]{let mut out=vec![0;n];e.pattern(&mut out,&seed);for i in 0..n{assert_eq!(out[i],seed[i%d]);}}
    }
}
#[test]
fn every_truncated_prefix_is_rejected(){
    let f=compress(&fixture(16_591,17),&Options{block_size:4096,..Options::default()}).unwrap();
    for n in 0..f.len(){assert!(decompress(&f[..n],&DecodeOptions::default()).is_err(),"prefix {n}");}
    let mut trailing=f.clone();trailing.push(0);assert!(decompress(&trailing,&DecodeOptions::default()).is_err());
}
#[test]
fn output_limit_and_checksum_are_enforced(){
    let data=fixture(100_000,0);let f=compress(&data,&Options::default()).unwrap();
    assert!(decompress(&f,&DecodeOptions{max_output:data.len()-1,..DecodeOptions::default()}).is_err());
    let mut corrupt=f;corrupt[32+24]^=1;
    assert!(matches!(decompress(&corrupt,&DecodeOptions::default()),Err(vxg::Error::Checksum{..})));
}
#[test]
fn block_random_access_is_independent(){
    let data=fixture(16_589,0);let opt=Options{block_size:4096,..Options::default()};let f=compress(&data,&opt).unwrap();
    let parsed=Frame::parse(&f,data.len()).unwrap();
    for index in (0..parsed.block_count()).rev(){assert_eq!(parsed.decode_block(index,&DecodeOptions::default()).unwrap(),data[index*4096..((index+1)*4096).min(data.len())]);}
}
#[test]
fn malformed_input_never_panics_under_bounded_allocation(){
    let options=DecodeOptions{max_output:1<<20,..DecodeOptions::default()};let mut seed=123456789;
    for n in 0..1024{let bytes:Vec<u8>=(0..n).map(|_|rng(&mut seed)).collect();assert!(std::panic::catch_unwind(||decompress(&bytes,&options)).is_ok());}
    let base=compress(&fixture(4096,17),&Options::default()).unwrap();
    for _ in 0..5000{let mut f=base.clone();let pos=(rng(&mut seed) as usize*257+rng(&mut seed) as usize)%f.len();f[pos]^=rng(&mut seed)|1;
        assert!(std::panic::catch_unwind(||decompress(&f,&options)).is_ok());}
}
#[test]
fn invalid_options_return_errors(){
    for bs in [0,1,4095,4097,1_048_577]{assert!(compress(b"abc",&Options{block_size:bs,..Options::default()}).is_err());}
    assert!(compress(b"",&Options{threads:0,..Options::default()}).is_err());
    assert!(compress(b"",&Options{acceleration:0,..Options::default()}).is_err());
}
#[cfg(not(any(feature="metal",feature="vulkan",feature="cuda")))]
#[test]
fn missing_gpu_backend_never_silently_falls_back(){
    for b in [vxg::gpu::Backend::Metal,vxg::gpu::Backend::Vulkan,vxg::gpu::Backend::Cuda]{assert!(vxg::gpu::GpuCodec::new(b).is_err());}
}
