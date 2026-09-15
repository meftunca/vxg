use std::{env,fs::{self,OpenOptions},io::Write,path::Path,process::ExitCode};
use vxg::{DecodeOptions,Options,Frame,Simd,gpu::{Backend,GpuCodec,GpuOptions}};
type R<T>=Result<T,Box<dyn std::error::Error>>;
const HELP:&str=r#"VXG 0.1 — experimental generic lossless codec

vxg c INPUT OUTPUT [--backend cpu|metal|vulkan|cuda] [--threads N]
                  [--block-size BYTES] [--acceleration 1..64]
                  [--simd auto|scalar|sse2|avx2|avx512|neon] [--no-checksum]
                  [--gpu-batch BYTES] [--force]
vxg d INPUT OUTPUT [--backend cpu|metal|vulkan|cuda] [--threads N]
                  [--simd MODE] [--max-output BYTES] [--gpu-batch BYTES] [--force]
vxg info FRAME
vxg hardware
vxg gpu-check metal|vulkan|cuda

Compression output is a new file unless --force is explicitly supplied.
GPU features require native SDKs at build time. No silent CPU fallback.
Numbers are bytes, not MiB. Default block 65536; GPU batch 16777216.
"#;
fn val<'a>(args:&'a[String],i:&mut usize)->R<&'a str>{*i+=1;Ok(args.get(*i).ok_or("missing option value")?.as_str())}
fn write(path:&str,bytes:&[u8],force:bool)->R<()>{
    let mut options=OpenOptions::new();options.write(true);
    if force{options.create(true).truncate(true);}else{options.create_new(true);}
    let mut f=options.open(path)?;f.write_all(bytes)?;f.flush()?;Ok(())
}
fn main()->ExitCode{match run(){Ok(())=>ExitCode::SUCCESS,Err(e)=>{eprintln!("error: {e}");ExitCode::FAILURE}}}
fn run()->R<()>{
    let args:Vec<String>=env::args().skip(1).collect();let Some(cmd)=args.first().map(String::as_str)else{print!("{HELP}");return Ok(());};
    match cmd {
        "--help"|"-h"|"help"=>{print!("{HELP}");return Ok(());}
        "hardware"=>{
            println!("os={} arch={} available_parallelism={}",env::consts::OS,env::consts::ARCH,std::thread::available_parallelism().map(|x|x.get()).unwrap_or(1));
            println!("auto SIMD={:?}",vxg::simd::Engine::new(Simd::Auto)?.kind());
            for s in [Simd::Scalar,Simd::Sse2,Simd::Avx2,Simd::Avx512,Simd::Neon]{println!("{s:?}: {}",vxg::simd::Engine::supported(s));}
            println!("compiled native features: metal={} vulkan={} cuda={}",cfg!(feature="metal"),cfg!(feature="vulkan"),cfg!(feature="cuda"));return Ok(());
        }
        "info"=>{
            if args.len()!=2{return Err("usage: vxg info FRAME".into());}
            let bytes=fs::read(&args[1])?;let f=Frame::parse(&bytes,usize::MAX)?;f.validate_commands()?;
            println!("raw={} encoded={} blocks={} block_size={} checksum={}",f.raw_size(),bytes.len(),f.block_count(),f.block_size(),f.checksummed());
            for(i,b)in f.blocks().enumerate(){println!("block={i} raw_offset={} raw={} payload_offset={} encoded={} kind={}",b.raw_offset,b.raw_size,b.payload_offset,b.payload_size,if b.kind==0{"RAW"}else{"LZ"});}return Ok(());
        }
        "gpu-check"=>{
            if args.len()!=2{return Err("usage: vxg gpu-check metal|vulkan|cuda".into());}
            return gpu_check(args[1].parse()?);
        }
        "c"|"d"=>{}
        _=>return Err(format!("unknown command: {cmd}\n{HELP}").into()),
    }
    if args.len()<3{return Err(HELP.into());}
    let input=&args[1];let output=&args[2];if input==output{return Err("input and output must differ".into());}
    let mut enc=Options::default();let mut dec=DecodeOptions::default();let mut gpu=GpuOptions::default();let mut backend="cpu".to_owned();let mut force=false;
    let mut i=3;
    while i<args.len(){
        match args[i].as_str(){
            "--backend"=>backend=val(&args,&mut i)?.into(),
            "--threads"=>{let n=val(&args,&mut i)?.parse()?;enc.threads=n;dec.threads=n;}
            "--block-size"=>{let n=val(&args,&mut i)?.parse()?;enc.block_size=n;gpu.block_size=n;}
            "--acceleration"=>enc.acceleration=val(&args,&mut i)?.parse()?,
            "--simd"=>{let s=val(&args,&mut i)?.parse()?;enc.simd=s;dec.simd=s;}
            "--no-checksum"=>{enc.checksum=false;gpu.checksum=false;}
            "--max-output"=>{let n=val(&args,&mut i)?.parse()?;dec.max_output=n;gpu.max_output=n;}
            "--gpu-batch"=>gpu.batch_bytes=val(&args,&mut i)?.parse()?,
            "--force"=>force=true,
            x=>return Err(format!("unknown option {x}").into()),
        }i+=1;
    }
    if !force&&Path::new(output).exists(){return Err("output exists (use --force to replace it)".into());}
    let bytes=fs::read(input)?;
    let start=std::time::Instant::now();
    let result=if backend=="cpu"{
        if cmd=="c"{vxg::compress(&bytes,&enc)?}else{vxg::decompress(&bytes,&dec)?}
    }else{
        if enc.acceleration!=1||enc.simd!=Simd::Auto||enc.threads!=1{return Err("--acceleration, --simd and --threads are CPU-only tuning options".into());}
        let mut device=GpuCodec::new(backend.parse()?)?;
        let (result,report)=if cmd=="c"{device.compress(&bytes,&gpu)?}else{device.decompress(&bytes,&gpu)?};
        eprintln!("device={} batches={} e2e_ms={:.3} kernel_ms={:?} (kernel time excludes transfers/host work)",report.device,report.batches,report.end_to_end_ms,report.kernel_ms);result
    };
    write(output,&result,force)?;
    eprintln!("{} -> {} bytes; {:.3} ms including codec setup and output write",bytes.len(),result.len(),start.elapsed().as_secs_f64()*1000.0);Ok(())
}
fn gpu_check(backend:Backend)->R<()>{
    let mut device=GpuCodec::new(backend)?;println!("device: {}",device.device_name());
    let gpu=GpuOptions{block_size:4096,batch_bytes:8192,..GpuOptions::default()};
    let cpu=Options{block_size:4096,..Options::default()};let mut count=0;
    for n in [0,1,7,31,127,4095,4096,4097,16391]{for period in [0,1,2,3,4,7,8,17,65]{
        let mut seed=0x123456789abcdefu64;
        let mut random=||{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;seed as u8};
        let pat:Vec<u8>=(0..period.max(1)).map(|_|random()).collect();
        let data:Vec<u8>=(0..n).map(|i|if period==0{random()}else{pat[i%period]}).collect();
        let cpu_frame=vxg::compress(&data,&cpu)?;
        let (gpu_frame,_)=device.compress(&data,&gpu)?;
        if vxg::decompress(&gpu_frame,&DecodeOptions::default())?!=data{return Err("GPU encoder -> CPU decoder mismatch".into());}
        if device.decompress(&cpu_frame,&gpu)?.0!=data{return Err("CPU encoder -> GPU decoder mismatch".into());}
        if device.decompress(&gpu_frame,&gpu)?.0!=data{return Err("GPU encoder -> GPU decoder mismatch".into());}
        count+=1;
    }}
    println!("PASS: {count} three-way CPU/GPU conformance cases");Ok(())
}
