use std::{env,fs,io::Write,hint::black_box,time::{Duration,Instant}};
use vxg::{compress_with_stats,Options,DecodeOptions,Frame,Simd,baselines::{self,Codec,Encoded},gpu::{Backend,GpuCodec,GpuOptions}};
type R<T>=Result<T,Box<dyn std::error::Error>>;
#[cfg(all(feature="mimalloc",feature="jemalloc"))]
compile_error!("benchmark allocator features are mutually exclusive");
#[cfg(any(all(feature="alloc-stats",feature="mimalloc"),all(feature="alloc-stats",feature="jemalloc")))]
compile_error!("alloc-stats cannot be combined with a replacement allocator");
#[cfg(all(feature="mimalloc",not(any(feature="jemalloc",feature="alloc-stats"))))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[cfg(all(feature="jemalloc",not(any(feature="mimalloc",feature="alloc-stats"))))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(feature="mimalloc",not(any(feature="jemalloc",feature="alloc-stats"))))]
fn allocator_name()->&'static str{"mimalloc"}
#[cfg(all(feature="jemalloc",not(any(feature="mimalloc",feature="alloc-stats"))))]
fn allocator_name()->&'static str{"jemalloc"}
#[cfg(feature="alloc-stats")]
fn allocator_name()->&'static str{"system+alloc-stats"}
#[cfg(not(any(feature="mimalloc",feature="jemalloc",feature="alloc-stats")))]
fn allocator_name()->&'static str{"system"}

#[cfg(feature="alloc-stats")]
mod alloc_probe {
    use std::alloc::{GlobalAlloc,Layout,System};
    use std::sync::atomic::{AtomicUsize,Ordering};

    pub struct CountingAllocator;
    static ALLOC_CALLS:AtomicUsize=AtomicUsize::new(0);
    static DEALLOC_CALLS:AtomicUsize=AtomicUsize::new(0);
    static REALLOC_CALLS:AtomicUsize=AtomicUsize::new(0);
    static ALLOC_BYTES:AtomicUsize=AtomicUsize::new(0);
    static DEALLOC_BYTES:AtomicUsize=AtomicUsize::new(0);

    #[global_allocator]
    pub static GLOBAL:CountingAllocator=CountingAllocator;
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self,layout:Layout)->*mut u8 {
            let ptr=System.alloc(layout);
            if !ptr.is_null(){ALLOC_CALLS.fetch_add(1,Ordering::Relaxed);ALLOC_BYTES.fetch_add(layout.size(),Ordering::Relaxed);}
            ptr
        }
        unsafe fn dealloc(&self,ptr:*mut u8,layout:Layout){DEALLOC_CALLS.fetch_add(1,Ordering::Relaxed);DEALLOC_BYTES.fetch_add(layout.size(),Ordering::Relaxed);System.dealloc(ptr,layout);}
        unsafe fn realloc(&self,ptr:*mut u8,layout:Layout,new_size:usize)->*mut u8 {
            let out=System.realloc(ptr,layout,new_size);
            if !out.is_null(){REALLOC_CALLS.fetch_add(1,Ordering::Relaxed);ALLOC_BYTES.fetch_add(new_size,Ordering::Relaxed);DEALLOC_BYTES.fetch_add(layout.size(),Ordering::Relaxed);}
            out
        }
    }
    #[derive(Clone,Copy)]
    pub struct Snapshot{pub alloc_calls:usize,pub dealloc_calls:usize,pub realloc_calls:usize,pub alloc_bytes:usize,pub dealloc_bytes:usize}
    pub fn reset(){for x in [&ALLOC_CALLS,&DEALLOC_CALLS,&REALLOC_CALLS,&ALLOC_BYTES,&DEALLOC_BYTES]{x.store(0,Ordering::Relaxed);}}
    pub fn snapshot()->Snapshot{Snapshot{alloc_calls:ALLOC_CALLS.load(Ordering::Relaxed),dealloc_calls:DEALLOC_CALLS.load(Ordering::Relaxed),realloc_calls:REALLOC_CALLS.load(Ordering::Relaxed),alloc_bytes:ALLOC_BYTES.load(Ordering::Relaxed),dealloc_bytes:DEALLOC_BYTES.load(Ordering::Relaxed)}}
}
#[derive(Clone)]
struct Config{size:usize,seconds:f64,samples:usize,threads:usize,block_size:usize,simd:Simd,csv:String,files:Vec<String>,gpu:Option<Backend>,microbench:bool,vxg_only:bool}
fn next<'a>(args:&'a[String],i:&mut usize)->R<&'a str>{*i+=1;Ok(args.get(*i).ok_or("missing value")?.as_str())}
fn main(){if let Err(e)=run(){eprintln!("benchmark failed: {e}");std::process::exit(1);}}
fn run()->R<()>{
    let mut c=Config{size:4<<20,seconds:0.1,samples:5,threads:1,block_size:65536,simd:Simd::Auto,csv:"reports/local-benchmark.csv".into(),files:Vec::new(),gpu:None,microbench:false,vxg_only:false};
    let args:Vec<String>=env::args().skip(1).collect();let mut i=0;
    while i<args.len(){match args[i].as_str(){
        "--size-mib"=>c.size=next(&args,&mut i)?.parse::<usize>()?.checked_mul(1<<20).ok_or("size overflow")?,
        "--seconds"=>c.seconds=next(&args,&mut i)?.parse()?,"--samples"=>c.samples=next(&args,&mut i)?.parse()?,
        "--threads"=>c.threads=next(&args,&mut i)?.parse()?,"--block-size"=>c.block_size=next(&args,&mut i)?.parse()?,
        "--simd"=>c.simd=next(&args,&mut i)?.parse()?,"--csv"=>c.csv=next(&args,&mut i)?.into(),
        "--gpu"=>c.gpu=Some(next(&args,&mut i)?.parse()?),"--file"=>c.files.push(next(&args,&mut i)?.into()),
        "--microbench"=>c.microbench=true,"--vxg-only"=>c.vxg_only=true,
        "--help"=>{println!("vxg-bench [--size-mib N] [--seconds 0.1] [--samples 5] [--threads 1] [--block-size 65536] [--simd auto] [--csv FILE] [--file CORPUS_FILE]... [--gpu metal|vulkan|cuda] [--microbench] [--vxg-only]\nAll CPU codecs use independent blocks, RAW fallback and no checksum. GPU rows are explicitly host-e2e, not CPU preallocated rows. --microbench measures Engine LCP/copy/pattern kernels without codec framing. --vxg-only skips baseline codecs and GPU rows.");return Ok(());}
        x=>return Err(format!("unknown option {x}").into()),
    }i+=1;}
    if c.size==0||!c.seconds.is_finite()||c.seconds<=0.0||c.seconds>60.0||c.samples<3||c.samples>101{return Err("invalid size, seconds (0..60], or samples (3..101)".into());}
    let engine=vxg::simd::Engine::new(c.simd)?;
    if c.microbench { return run_microbench(&c, engine); }
    #[cfg(feature="alloc-stats")]
    eprintln!("ALLOC-STATS enabled: allocation counters perturb timing; use counts only, not GB/s.");
    let (lz,zs)=baselines::versions();
    eprintln!("VXG 0.1 / liblz4 {lz} / libzstd {zs}; {} {}; SIMD={:?}; allocator={}; threads={}; block={}",env::consts::OS,env::consts::ARCH,engine.kind(),allocator_name(),c.threads,c.block_size);
    eprintln!("CPU: allocating encode, prepared-directory/preallocated decode. Per-call contexts included. RAW fallback for ALL codecs. Checksums OFF. Warm cache. GPU: separate host-e2e profile.");
    let mut datasets=if c.files.is_empty(){datasets(c.size)}else{Vec::new()};
    for path in &c.files{datasets.push((path.clone(),fs::read(path)?));}
    if let Some(parent)=std::path::Path::new(&c.csv).parent(){if !parent.as_os_str().is_empty(){fs::create_dir_all(parent)?;}}
    let mut csv=fs::File::create(&c.csv)?;
    writeln!(csv,"dataset,codec,profile,operation,raw_bytes,encoded_bytes,ratio_encoded_over_raw,threads,block_size,median_ns,min_ns,max_ns,median_GBps,samples,checksums,lz4_version,zstd_version,device,allocator")?;
    let mut gpu=match c.gpu{Some(b)=>Some(GpuCodec::new(b)?),None=>None};
    for (name,data) in datasets{
        if data.is_empty(){eprintln!("skip empty dataset {name}");continue;}
        let enc_opt=Options{block_size:c.block_size,threads:c.threads,checksum:false,simd:c.simd,..Options::default()};
        let dec_opt=DecodeOptions{threads:c.threads,simd:c.simd,max_output:data.len(),verify_checksum:false};
        #[cfg(feature="alloc-stats")]
        alloc_probe::reset();
        let (frame,work_stats)=compress_with_stats(&data,&enc_opt)?;
        #[cfg(feature="alloc-stats")]
        { let a=alloc_probe::snapshot(); eprintln!("{name:22} alloc calls={} realloc={} dealloc={} alloc_bytes={} dealloc_bytes={} live_bytes={}",a.alloc_calls,a.realloc_calls,a.dealloc_calls,a.alloc_bytes,a.dealloc_bytes,a.alloc_bytes.saturating_sub(a.dealloc_bytes)); }
        let parsed=Frame::parse(&frame,data.len())?;let mut out=vec![0u8;data.len()];
        eprintln!("{name:22} VXG work raw_fast={} sample_words={} probes={} occupied={} candidate_checks={} collisions={} lcp_calls={} lcp_bytes={} matches={} matched_bytes={} streams=control:{} attrs:{} literals:{} ops={:?}",work_stats.raw_fast_path_blocks,work_stats.raw_sample_words,work_stats.probes,work_stats.occupied_probes,work_stats.candidate_checks,work_stats.hash_collisions,work_stats.lcp_calls,work_stats.lcp_bytes,work_stats.accepted_matches,work_stats.matched_bytes,work_stats.control_bytes,work_stats.attribute_bytes,work_stats.literal_bytes,work_stats.op_counts);
        parsed.decode_into(&mut out,&dec_opt)?;assert_eq!(out,data,"VXG correctness before timing");
        let en=measure(&c,||{let f=vxg::compress(black_box(&data),&enc_opt)?;black_box(f);Ok(())})?;
        row(&mut csv,&c,&name,"vxg","cpu_allocating_encode","compress",data.len(),frame.len(),&en,&lz,&zs,"")?;
        let de=measure(&c,||{parsed.decode_into(black_box(&mut out),&dec_opt)?;black_box(&out);Ok(())})?;
        assert_eq!(out,data,"VXG correctness after timing");
        row(&mut csv,&c,&name,"vxg","cpu_preallocated_decode","decompress",data.len(),frame.len(),&de,&lz,&zs,"")?;
        if c.vxg_only { csv.flush()?; continue; }
        for codec in [Codec::Lz4{acceleration:1},Codec::Lz4{acceleration:4},Codec::Zstd{level:-3},Codec::Zstd{level:1},Codec::Zstd{level:3}]{
            let compressed=Encoded::compress(&data,codec,c.block_size,c.threads)?;
            compressed.decode_into(&mut out,c.threads)?;assert_eq!(out,data,"baseline correctness before timing");
            let en=measure(&c,||{let f=Encoded::compress(black_box(&data),codec,c.block_size,c.threads)?;black_box(f);Ok(())})?;
            row(&mut csv,&c,&name,&codec.name(),"cpu_allocating_encode","compress",data.len(),compressed.encoded_size(),&en,&lz,&zs,"")?;
            let de=measure(&c,||{compressed.decode_into(black_box(&mut out),c.threads)?;black_box(&out);Ok(())})?;
            assert_eq!(out,data,"baseline correctness after timing");
            row(&mut csv,&c,&name,&codec.name(),"cpu_preallocated_decode","decompress",data.len(),compressed.encoded_size(),&de,&lz,&zs,"")?;
        }
        if let Some(device)=gpu.as_mut(){
            let options=GpuOptions{block_size:c.block_size,checksum:false,max_output:data.len(),..GpuOptions::default()};
            let (compressed,_)=device.compress(&data,&options)?;
            assert_eq!(vxg::decompress(&compressed,&dec_opt)?,data,"GPU encode / CPU decode correctness");
            assert_eq!(device.decompress(&frame,&options)?.0,data,"CPU encode / GPU decode correctness");
            let device_name=device.device_name().to_owned();
            let en=measure(&c,||{black_box(device.compress(black_box(&data),&options)?);Ok(())})?;
            row(&mut csv,&c,&name,"vxg-gpu","gpu_host_e2e_allocating","compress",data.len(),compressed.len(),&en,&lz,&zs,&device_name)?;
            let de=measure(&c,||{black_box(device.decompress(black_box(&compressed),&options)?);Ok(())})?;
            row(&mut csv,&c,&name,"vxg-gpu","gpu_host_e2e_allocating","decompress",data.len(),compressed.len(),&de,&lz,&zs,&device_name)?;
            let (decoded,report)=device.decompress(&compressed,&options)?;assert_eq!(decoded,data);
            eprintln!("GPU device timestamp observation (not median, not e2e): {} kernel_ms={:?}, host_e2e_ms={:.3}",name,report.kernel_ms,report.end_to_end_ms);
        }
        csv.flush()?;
    }
    eprintln!("CSV: {}",c.csv);Ok(())
}
fn run_microbench(c:&Config,engine:vxg::simd::Engine)->R<()>{
    let sizes=[8usize,16,32,64,256,4096,65536,1<<20];
    println!("operation,simd,bytes,median_ns,min_ns,max_ns,GBps,samples");
    for &n in &sizes{
        let a=vec![0x5au8;n];let b=vec![0x5au8;n];
        let (median,min,max)=micro_measure(c,||{black_box(engine.lcp(black_box(&a),black_box(&b)));});
        println!("lcp-equal,{:?},{n},{median:.3},{min:.3},{max:.3},{:.4},{}",engine.kind(),n as f64/median,c.samples);

        let src=vec![0x3cu8;n];let mut dst=vec![0u8;n];
        let (median,min,max)=micro_measure(c,||{engine.copy(black_box(&mut dst),black_box(&src));black_box(dst.as_ptr());});
        println!("copy,{:?},{n},{median:.3},{min:.3},{max:.3},{:.4},{}",engine.kind(),n as f64/median,c.samples);
    }
    let seed:Vec<u8>=(0..37).map(|x|x as u8).collect();let mut dst=vec![0u8;1<<20];
    let (median,min,max)=micro_measure(c,||{engine.pattern(black_box(&mut dst),black_box(&seed));black_box(dst.as_ptr());});
    println!("pattern-period-37,{:?},{},{median:.3},{min:.3},{max:.3},{:.4},{}",engine.kind(),dst.len(),dst.len() as f64/median,c.samples);
    let mut dst=vec![0u8;1<<20];
    let (median,min,max)=micro_measure(c,||{platform_pattern(black_box(&mut dst),black_box(&seed));black_box(dst.as_ptr());});
    println!("pattern-platform-period-37,{:?},{},{median:.3},{min:.3},{max:.3},{:.4},{}",engine.kind(),dst.len(),dst.len() as f64/median,c.samples);
    Ok(())
}
fn platform_pattern(dst:&mut [u8],seed:&[u8]){
    if seed.len()==1{dst.fill(seed[0]);return;}
    let step=(1024/seed.len())*seed.len();let mut tile=[0u8;1024];
    tile[..seed.len()].copy_from_slice(seed);let mut filled=seed.len();
    while filled<step{let take=filled.min(step-filled);let (prefix,suffix)=tile.split_at_mut(filled);suffix[..take].copy_from_slice(&prefix[..take]);filled+=take;}
    for chunk in dst.chunks_mut(step){chunk.copy_from_slice(&tile[..chunk.len()]);}
}
fn micro_measure<F>(c:&Config,mut f:F)->(f64,f64,f64) where F:FnMut(){
    let mut samples=Vec::with_capacity(c.samples);let duration=Duration::from_secs_f64(c.seconds);
    for _ in 0..c.samples{
        let start=Instant::now();let mut ops=0u64;
        while start.elapsed()<duration{f();ops+=1;}
        samples.push(start.elapsed().as_secs_f64()*1e9/ops.max(1) as f64);
    }
    samples.sort_by(f64::total_cmp);let mid=samples.len()/2;
    let median=if samples.len()%2==0{(samples[mid-1]+samples[mid])*0.5}else{samples[mid]};
    (median,samples[0],samples[samples.len()-1])
}
fn measure<F>(c:&Config,mut f:F)->R<Vec<f64>> where F:FnMut()->R<()>{
    f()?;f()?; // Warm-up, never included.
    let mut samples=Vec::with_capacity(c.samples);let duration=Duration::from_secs_f64(c.seconds);
    for _ in 0..c.samples{
        let start=Instant::now();let mut n=0u64;
        loop{f()?;n+=1;if start.elapsed()>=duration{break;}}
        samples.push(start.elapsed().as_secs_f64()*1e9/n as f64);
    }
    samples.sort_by(f64::total_cmp);Ok(samples)
}
fn quote(s:&str)->String{format!("\"{}\"",s.replace('"',"\"\""))}
#[allow(clippy::too_many_arguments)]
fn row(out:&mut fs::File,c:&Config,name:&str,codec:&str,profile:&str,op:&str,n:usize,m:usize,s:&[f64],lz:&str,zs:&str,device:&str)->R<()>{
    let mid=s.len()/2;let median=if s.len()%2==0{(s[mid-1]+s[mid])*0.5}else{s[mid]};let gbps=n as f64/median;
    writeln!(out,"{},{},{},{},{},{},{:.8},{},{},{:.3},{:.3},{:.3},{:.4},{},off,{},{},{},{}",quote(name),codec,profile,op,n,m,m as f64/n as f64,c.threads,c.block_size,median,s[0],s[s.len()-1],gbps,s.len(),lz,zs,quote(device),allocator_name())?;
    eprintln!("{name:22} {codec:14} {op:10} ratio={:.3} {:8.3} GB/s [{profile}]",m as f64/n as f64,gbps);Ok(())
}
fn rng(s:&mut u64)->u8{*s^=*s<<13;*s^=*s>>7;*s^=*s<<17;*s as u8}
fn datasets(n:usize)->Vec<(String,Vec<u8>)>{
    let mut seed=0x123456789abcdefu64;
    let random:Vec<u8>=(0..n).map(|_|rng(&mut seed)).collect();
    let low:Vec<u8>=(0..n).map(|_|rng(&mut seed)&15).collect();
    let period:Vec<u8>=(0..n).map(|i|((i%37)*7+3) as u8).collect();
    let mut json=Vec::with_capacity(n);let mut i=0;
    while json.len()<n{json.extend_from_slice(format!("{{\"tenant\":{},\"event\":\"checklist.completed\",\"item\":{},\"ok\":true,\"actor\":\"user-{}\"}}\n",i%31,i,i%103).as_bytes());i+=1;}json.truncate(n);
    let mixed:Vec<u8>=(0..n).map(|i|match(i/4096)%4{0=>0,1=>random[i],2=>json[i],_=>period[i]}).collect();
    vec![("zeros".into(),vec![0;n]),("period-37".into(),period),("json-records".into(),json),("random".into(),random),("low-entropy-random".into(),low),("mixed-4KiB".into(),mixed)]
}
