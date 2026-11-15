//! Disjoint multi-domain pack-time calibration corpus for `VXT-Turbo v3`.
//! Zero 8-word overlap with WikiText-2 or `TR+Code+Math` evaluation suites.

#[cfg(test)]
use std::collections::HashSet;

use crate::tokenizer::BpeTokenizer;

pub(crate) const CALIB_PASSAGES: [&str; 16] = [
    // 0..4: English systems & physical science
    "Geothermal binary-cycle power plants tapping hypersaline brines encounter rapid amorphous silica polymerization whenever temperature drops below quartz saturation inside titanium plate heat exchangers. Operators mitigate colloidal deposition by dosing hydrochloric acid upstream to depress hydroxide-catalyzed condensation kinetics, extending reinjection well permeability across multi-year campaigns while monitoring differential pressure drops across staggered chevron channels and acoustic scaling probes. Transient fouling correlations furthermore incorporate brine ionic strength, dwell-time distributions, and wall-shear erosion coefficients across seasonal electrical load swings.",
    "Autonomous bathymetric vehicles mapping steep continental-slope canyons combine multibeam echosounders with fiber-optic inertial navigation and phased-array Doppler velocity logs. Because acoustic ray bending varies sharply across internal-wave pycnoclines, onboard Kalman smoothers continuously invert expandable bathythermograph profiles alongside overlap swath tie-lines to eliminate outer-beam roll artifacts and maintain sub-decimeter vertical repeatability. Residual tidal phase lags near submarine fan levees are resolved by blending pressure-gauge bottom nodes with dense swath cross-over constraints during mission post-processing.",
    "Hafnium-zirconium oxide ferroelectric capacitors embedded in nonvolatile crossbar arrays exhibit wake-up polarization growth followed by fatigue as dipolar domains pin near grain boundaries under bipolar pulses. Pulsed capacitance-voltage spectroscopy shows oxygen vacancies drifting from bottom titanium-nitride electrodes create internal bias fields that shift coercive switching thresholds, requiring asymmetric reset waveforms and rapid thermal annealing. Finite-element electrothermal modeling confirms localized filament heating accelerates grain-boundary charge trapping unless wordline rise times remain capped below five nanoseconds.",
    "Cryogenic ammonia fuel bunkering aboard ocean-going cargo vessels demands continuous boil-off vapor recovery through cascade reliquefaction skids before purging insulated manifold arms. Dynamic relief-valve simulations couple homogeneous equilibrium two-phase pipe flow with tank headspace stratification so sudden pump trips never induce destructive liquid hammer inside austenitic stainless steel vacuum-jacketed transfer lines. Inert nitrogen sweep cycles verify sub-ppm vapor clearance prior to quick-disconnect coupler separation under port bunkering checklists.",
    // 4..8: Turkish agglutinative scientific & engineering prose
    "Marmara Denizi'nin iki tabakalı akıntı sisteminde Karadeniz kaynaklı az tuzlu üst su kütlesi ile Akdeniz kökenli tuzlu alt tabaka arasındaki keskin yoğunluk sınırı dikey karışımı büyük ölçüde sınırlandırmaktadır. Bahar aylarında artan fitoplankton birincil üretiminin dibe çökerek bakteriyel ayrışmaya uğraması, termoklin altındaki çözünmüş oksijen derişimini kritik eşiklerin altına düşürmekte ve bentik omurgasız topluluklarının dikey dağılımını daraltmaktadır. Kıyısal oşinografi şamandıralarından aktarılan gerçek zamanlı tuzluluk ve bulanıklık profilleri, oksijen yenilenme hızlarının mevsimsel modellemesini doğrular.",
    "Kütahya ve İznik geleneğindeki kuvars ağırlıklı çini hamurlarının yüksek sıcaklık fırınlaması sırasında oluşan camsı fazın genleşme katsayısı, alt astar ile sır tabakası arasındaki mekanik uyumu belirlemektedir. Fırın atmosferinin oksitleyici evreden hafif indirgeyici evreye kontrollü biçimde geçirilmesi, kobalt ve bakır iyonlarının değerlik durumunu değiştirerek yüzeyde mikroskobik çatlak oluşmadan derin turkuaz ve mercan tonlarının vitrifikasyonunu güvence altına alır. Soğutma eğrisinin kritik kuvars dönüşüm sıcaklığı çevresinde yavaşlatılması ise iç gerilmeleri tamamen sönümlemektedir.",
    "Güneydoğu Anadolu havzalarındaki açık sulama kanallarında yaz kuraklığı boyunca yaşanan yüksek buharlaşma kayıpları hem su bütçesini zorlamakta hem de kılcal yükselme yoluyla tarım topraklarında ikincil tuzlulaşmayı hızlandırmaktadır. Toprak nem tansiyometrelerinden gelen telemetri verileriyle yönlendirilen basınç dengelemeli damla sulama otomasyonu, kök bölgesi tuz birikimini yıkama eşiğinin altında tutarken birim sulama suyu verimliliğini belirgin biçimde yükseltmektedir. Gece saatlerinde gerçekleştirilen darbeli sulama programları yaprak yüzeyinden kaybolan suyu azaltarak pamuk ve mısır verim kararlılığını destekler.",
    "Yerli rüzgar türbini kanatlarının kök bağlantılarından uç profillerine uzanan karbon-epoksi kompozit kabukları, yirmi yıllık işletme ömrü boyunca yüz milyonlarca çevrimlik değişken aerodinamik bükülme yüküne dayanmak zorundadır. Uç kısımlara yerleştirilen alüminyum yıldırım yakalama reseptörleri ile iç kiriş boyunca uzanan düşük empedanslı bakır iniş iletkenlerinin termal genleşme uyumu, yüksek akımlı boşalmalarda delaminasyon riskini ortadan kaldırmak için sonlu elemanlar analizleriyle doğrulanır. Gövde içine yerleştirilen optik lifli gerinim sensörleri buzlanma kaynaklı dengesizlikleri erken aşamada saptar.",
    // 8..12: Polyglot source code & database queries (C99, PostgreSQL, TypeScript, Go)
    "/* Lock-free SPSC telemetry ring buffer with tableless Castagnoli CRC-32C folding */\nstatic inline uint32_t crc32c_step(uint32_t crc, const uint8_t *buf, size_t len) {\n    crc = ~crc;\n    for (size_t i = 0; i < len; ++i) {\n        crc ^= buf[i];\n        for (int b = 0; b < 8; ++b) {\n            uint32_t mask = (uint32_t)-(int32_t)(crc & 1u);\n            crc = (crc >> 1) ^ (0x82F63B78u & mask);\n        }\n    }\n    return ~crc;\n}\nbool ring_try_push(spsc_ring_t *r, const frame_hdr_t *hdr, const uint8_t *payload) {\n    size_t head = atomic_load_explicit(&r->head, memory_order_relaxed);\n    size_t tail = atomic_load_explicit(&r->tail, memory_order_acquire);\n    if (head - tail >= r->capacity) { r->drops++; return false; }\n    size_t slot = head & (r->capacity - 1u);\n    r->slots[slot].crc = crc32c_step(0u, payload, hdr->bytes);\n    atomic_store_explicit(&r->head, head + 1u, memory_order_release);\n    return true;\n}",
    "-- Cold-chain warehouse sensor excursion detection via PostgreSQL window frames\nWITH lagged_readings AS (\n    SELECT pallet_id, chamber_zone, recorded_at, temp_celsius,\n           LAG(temp_celsius) OVER (PARTITION BY pallet_id ORDER BY recorded_at) AS prev_temp,\n           EXTRACT(EPOCH FROM recorded_at - LAG(recorded_at) OVER (PARTITION BY pallet_id ORDER BY recorded_at)) AS dt_sec\n    FROM warehouse_telemetry\n    WHERE recorded_at >= NOW() - INTERVAL '14 days'\n)\nSELECT pallet_id, chamber_zone, COUNT(*) FILTER (WHERE temp_celsius > -17.5 AND prev_temp > -17.5) AS breaches,\n       ROUND(SUM(GREATEST(temp_celsius + 18.0, 0.0) * COALESCE(dt_sec, 0.0))::numeric, 2) AS degree_seconds\nFROM lagged_readings\nGROUP BY pallet_id, chamber_zone\nHAVING MAX(temp_celsius) > -16.8\nORDER BY degree_seconds DESC, breaches DESC;",
    "// WebGPU pipeline bind-group cache with deterministic discriminant key hashing\ntype BufferUsageTag = 'storage-ro' | 'storage-rw' | 'uniform';\ninterface BindingEntry { readonly slot: number; readonly tag: BufferUsageTag; readonly byteLen: number; }\nexport class BindGroupCache {\n    private readonly cache = new Map<string, GPUBindGroup>();\n    constructor(private readonly device: GPUDevice) {}\n    public getOrCreate(layout: GPUBindGroupLayout, entries: readonly BindingEntry[], buffers: readonly GPUBuffer[]): GPUBindGroup {\n        const key = entries.map((e, i) => `${e.slot}:${e.tag}:${e.byteLen}:${i}`).join('|');\n        const hit = this.cache.get(key);\n        if (hit !== undefined) return hit;\n        const bg = this.device.createBindGroup({\n            layout,\n            entries: entries.map((e, idx) => ({ binding: e.slot, resource: { buffer: buffers[idx], size: e.byteLen } })),\n        });\n        this.cache.set(key, bg);\n        return bg;\n    }\n}",
    "// Bounded worker pool with token-bucket pacing and graceful context draining\npackage scheduler\nimport (\n    \"context\"\n    \"sync\"\n    \"time\"\n)\nfunc RunBatch(ctx context.Context, workers int, rate time.Duration, jobs <-chan func() error) error {\n    ticker := time.NewTicker(rate)\n    defer ticker.Stop()\n    var wg sync.WaitGroup\n    errCh := make(chan error, 1)\n    for i := 0; i < workers; i++ {\n        wg.Add(1)\n        go func() {\n            defer wg.Done()\n            for j := range jobs {\n                select {\n                case <-ctx.Done():\n                    return\n                case <-ticker.C:\n                }\n                if err := j(); err != nil {\n                    select { case errCh <- err: default: }\n                }\n            }\n        }()\n    }\n    wg.Wait()\n    select { case e := <-errCh: return e; default: return nil }\n}",
    // 12..16: Mathematics & mathematical physics derivations
    "Consider the radial Helmholtz equation \\((\\partial_r^2 + r^{-1}\\partial_r + k^2 - \\nu^2 r^{-2}) G_\\nu(r,r') = -r^{-1}\\delta(r-r')\\) subject to mixed Robin impedance boundary conditions \\(r\\partial_r G + \\zeta G = 0\\) at inner cylinder radius \\(r = a\\) and Sommerfeld outgoing radiation as \\(r \\to \\infty\\). Matching the Wronskian \\(W[J_\\nu, H_\\nu^{(1)}](kr') = 2i/(\\pi kr')\\) yields the exact spectral Green kernel in terms of Bessel-Hankel ratios whose poles locate creeping surface modes. Deforming the contour around the branch cut isolates continuous radiation spectra from exponentially decaying leaky circumferential waves.",
    "For a non-autonomous Hamiltonian \\(H(q, p, \\varepsilon t) = \\frac{1}{2}p^2 + V(q, \\varepsilon t)\\) drifting slowly over timescale \\(\\varepsilon^{-1}\\), a fourth-order Yoshida symmetric splitting preserves the extended Poincaré-Cartan two-form \\(dp \\wedge dq - dH \\wedge dt\\) up to exponentially small remainders. Consequently the classical action integral \\(I = (2\\pi)^{-1}\\oint p\\,dq\\) stays adiabatically constant across resonant separatrix crossings modulo subleading logarithmic phase jumps. Matched asymptotic expansions along homoclinic manifolds predict capture probabilities purely from area-flux ratios evaluated at frozen parameter values.",
    "On a compact oriented smooth four-manifold \\((M^4, g)\\) with Levi-Civita curvature two-form \\(\\Omega^i{}_j\\), the Chern-Gauss-Bonnet integrand equals the Pfaffian \\(\\mathrm{Pf}(\\Omega) = (32\\pi^2)^{-1}(|\\mathrm{Riem}|^2 - 4|\\mathrm{Ric}|^2 + R^2)\\,dV_g\\). Decomposing the Riemann tensor into Weyl self-dual and anti-self-dual pieces together with trace-free Ricci curvature proves Hirzebruch's signature bound \\(2\\chi(M) \\ge 3|\\tau(M)|\\) whenever Einstein metrics satisfy \\(\\mathrm{Ric} = \\lambda g\\). Saturation of equality forces vanishing negative Weyl curvature and reduces local holonomy to special unitary subgroups.",
    "Let \\(S_n = \\sum_{k=1}^n X_k\\) be additive rewards accumulated along an irreducible finite-state Markov chain with transition kernel \\(P\\). tilting the generator via the spectral radius \\(\\rho(\\theta)\\) of \\(P_\\theta(x,y) = P(x,y)\\exp(\\theta f(y))\\) gives the scaled cumulant generating function \\(\\Lambda(\\theta) = \\ln \\rho(\\theta)\\). Its Legendre-Fenchel dual \\(\\Lambda^*(a) = \\sup_\\theta(\\theta a - \\Lambda(\\theta))\\) governs exact exponential tail decay of empirical buffer overflow probabilities. Gartner-Ellis large-deviation upper and lower bounds converge whenever the tilted Perron eigenvector remains strictly positive on every recurrent communicating class.",
];

#[cfg(test)]
fn normalize_words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_ascii_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Counts how many `n`-word n-grams in `b` appear in `a`. Returns `0` when inputs are shorter than `n`.
#[cfg(test)]
pub(crate) fn check_max_ngram_overlap(a: &str, b: &str, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let wa = normalize_words(a);
    let wb = normalize_words(b);
    if wa.len() < n || wb.len() < n {
        return 0;
    }
    let set: HashSet<&[String]> = wa.windows(n).collect();
    wb.windows(n).filter(|win| set.contains(win)).count()
}

/// Tokenizes custom text (or the 16 built-in disjoint passages) into fixed-length calibration sequences.
pub(crate) fn build_calib_sequences(
    tok: &BpeTokenizer,
    custom_text: Option<&str>,
    seq_len: usize,
    max_seqs: usize,
) -> Vec<Vec<u32>> {
    if seq_len == 0 || max_seqs == 0 || tok.vocab_size() == 0 {
        return Vec::new();
    }
    let pad_to_len = |ids: &[u32]| -> Option<Vec<u32>> {
        if ids.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(seq_len);
        while out.len() < seq_len {
            let take = (seq_len - out.len()).min(ids.len());
            out.extend_from_slice(&ids[..take]);
        }
        Some(out)
    };
    if let Some(txt) = custom_text.map(str::trim).filter(|s| !s.is_empty()) {
        let ids = tok.encode(txt);
        let mut seqs: Vec<Vec<u32>> = ids
            .chunks_exact(seq_len)
            .take(max_seqs)
            .map(|c| c.to_vec())
            .collect();
        if seqs.is_empty() {
            if let Some(s) = pad_to_len(&ids) {
                seqs.push(s);
            }
        }
        if !seqs.is_empty() {
            return seqs;
        }
    }
    CALIB_PASSAGES
        .iter()
        .take(max_seqs)
        .filter_map(|p| {
            let ids = tok.encode(p);
            if ids.len() >= seq_len {
                Some(ids[..seq_len].to_vec())
            } else {
                pad_to_len(&ids)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{build_calib_sequences, check_max_ngram_overlap, CALIB_PASSAGES};
    use crate::tokenizer::BpeTokenizer;

    fn mock_tokenizer() -> BpeTokenizer {
        let mut vocab = vec!["<unk>".to_owned(), " ".to_owned(), "\n".to_owned(), "\t".to_owned()];
        for b in 32u8..=126 {
            let ch = (b as char).to_string();
            if !vocab.contains(&ch) {
                vocab.push(ch);
            }
        }
        for tok in ["th", "he", "in", "er", "on", "an", "re", "ed", "es", "en", "ar", "st"] {
            vocab.push(tok.to_owned());
        }
        BpeTokenizer::from_vocab_and_merges(vocab, &[], None).unwrap()
    }

    #[test]
    fn builds_sixteen_sequences_of_requested_length() {
        let tok = mock_tokenizer();
        let seqs = build_calib_sequences(&tok, None, 128, 16);
        assert_eq!(seqs.len(), 16);
        assert!(seqs.iter().all(|s| s.len() == 128));
    }

    #[test]
    fn zero_shared_eight_word_ngrams_across_passages_and_eval_samples() {
        for (i, a) in CALIB_PASSAGES.iter().enumerate() {
            for (j, b) in CALIB_PASSAGES.iter().enumerate() {
                if i != j {
                    assert_eq!(check_max_ngram_overlap(a, b, 8), 0, "pair {i}-{j}");
                }
            }
        }
        let all = CALIB_PASSAGES.join("\n");
        let eval_probe = "Valkyria Chronicles III Senjō no Valkyria 3 unrecorded chronicles Türkiye ekonomisi tarımsal üretim endeksi function binarySearch arr target fn main";
        assert_eq!(check_max_ngram_overlap(&all, eval_probe, 8), 0);
        assert!(check_max_ngram_overlap("a b c d e f g h i", "x a b c d e f g h y", 8) > 0);
    }
}
