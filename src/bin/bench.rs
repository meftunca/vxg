use std::{env,fs,io::Write,hint::black_box,time::{Duration,Instant}};
use vxg::{Options,DecodeOptions,Frame,Simd,baselines::{self,Codec,Encoded},gpu::{Backend,GpuCodec,GpuOptions}};
type R<T>=Result<T,Box<dyn std::error::Error>>;
#[derive(Clone)]
struct Config{size:usize,seconds:f64,samples:usize,threads:usize,block_size:usize,simd:Simd,csv:String,files:Vec<String>,gpu:Option<Backend>}
fn next<'a>(args:&'a[String],i:&mut usize)->R<&'a str>{*i+=1;Ok(args.get(*i).ok_or("missing value")?.as_str())}
fn main(){if let Err(e)=run(){eprintln!("benchmark failed: {e}");std::process::exit(1);}}
fn run()->R<()>{
    let mut c=Config{size:4<<20,seconds:0.1,samples:5,threads:1,block_size:65536,simd:Simd::Auto,csv:"reports/local-benchmark.csv".into(),files:Vec::new(),gpu:None};
    let args:Vec<String>=env::args().skip(1).collect();let mut i=0;
    while i<args.len(){match args[i].as_str(){
        "--size-mib"=>c.size=next(&args,&mut i)?.parse::<usize>()?.checked_mul(1<<20).ok_or("size overflow")?,
        "--seconds"=>c.seconds=next(&args,&mut i)?.parse()?,"--samples"=>c.samples=next(&args,&mut i)?.parse()?,
        "--threads"=>c.threads=next(&args,&mut i)?.parse()?,"--block-size"=>c.block_size=next(&args,&mut i)?.parse()?,
        "--simd"=>c.simd=next(&args,&mut i)?.parse()?,"--csv"=>c.csv=next(&args,&mut i)?.into(),
        "--gpu"=>c.gpu=Some(next(&args,&mut i)?.parse()?),"--file"=>c.files.push(next(&args,&mut i)?.into()),
        "--help"=>{println!("vxg-bench [--size-mib N] [--seconds 0.1] [--samples 5] [--threads 1] [--block-size 65536] [--simd auto] [--csv FILE] [--file CORPUS_FILE]... [--gpu metal|vulkan|cuda]\nAll CPU codecs use independent blocks, RAW fallback and no checksum. GPU rows are explicitly host-e2e, not CPU preallocated rows.");return Ok(());}
        x=>return Err(format!("unknown option {x}").into()),
    }i+=1;}
    if c.size==0||!c.seconds.is_finite()||c.seconds<=0.0||c.seconds>60.0||c.samples<3||c.samples>101{return Err("invalid size, seconds (0..60], or samples (3..101)".into());}
    let (lz,zs)=baselines::versions();
    eprintln!("VXG 0.1 / liblz4 {lz} / libzstd {zs}; {} {}; SIMD={:?}; threads={}; block={}",env::consts::OS,env::consts::ARCH,vxg::simd::Engine::new(c.simd)?.kind(),c.threads,c.block_size);
    eprintln!("CPU: allocating encode, prepared-directory/preallocated decode. Per-call contexts included. RAW fallback for ALL codecs. Checksums OFF. Warm cache. GPU: separate host-e2e profile.");
    let mut datasets=if c.files.is_empty(){datasets(c.size)}else{Vec::new()};
    for path in &c.files{datasets.push((path.clone(),fs::read(path)?));}
    if let Some(parent)=std::path::Path::new(&c.csv).parent(){if !parent.as_os_str().is_empty(){fs::create_dir_all(parent)?;}}
    let mut csv=fs::File::create(&c.csv)?;
    writeln!(csv,"dataset,codec,profile,operation,raw_bytes,encoded_bytes,ratio_encoded_over_raw,threads,block_size,median_ns,min_ns,max_ns,median_GBps,samples,checksums,lz4_version,zstd_version,device")?;
    let mut gpu=match c.gpu{Some(b)=>Some(GpuCodec::new(b)?),None=>None};
    for (name,data) in datasets{
        if data.is_empty(){eprintln!("skip empty dataset {name}");continue;}
        let enc_opt=Options{block_size:c.block_size,threads:c.threads,checksum:false,simd:c.simd,..Options::default()};
        let dec_opt=DecodeOptions{threads:c.threads,simd:c.simd,max_output:data.len(),verify_checksum:false};
        let frame=vxg::compress(&data,&enc_opt)?;let parsed=Frame::parse(&frame,data.len())?;let mut out=vec![0u8;data.len()];
        parsed.decode_into(&mut out,&dec_opt)?;assert_eq!(out,data,"VXG correctness before timing");
        let en=measure(&c,||{let f=vxg::compress(black_box(&data),&enc_opt)?;black_box(f);Ok(())})?;
        row(&mut csv,&c,&name,"vxg","cpu_allocating_encode","compress",data.len(),frame.len(),&en,&lz,&zs,"")?;
        let de=measure(&c,||{parsed.decode_into(black_box(&mut out),&dec_opt)?;black_box(&out);Ok(())})?;
        assert_eq!(out,data,"VXG correctness after timing");
        row(&mut csv,&c,&name,"vxg","cpu_preallocated_decode","decompress",data.len(),frame.len(),&de,&lz,&zs,"")?;
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
    writeln!(out,"{},{},{},{},{},{},{:.8},{},{},{:.3},{:.3},{:.3},{:.4},{},off,{},{},{}",quote(name),codec,profile,op,n,m,m as f64/n as f64,c.threads,c.block_size,median,s[0],s[s.len()-1],gbps,s.len(),lz,zs,quote(device))?;
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
