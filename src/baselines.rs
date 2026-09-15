//! Bench-only C reference implementations. Link real liblz4/libzstd, not surrogate codecs.
//! Every codec gets the same independent block size and worker count. Encoded sizes
//! include the same 32 + 32*block_count accounting envelope; it is not an LZ4 frame.
use std::{ffi::{c_char,c_int,c_void,CStr}, ptr::NonNull};
use crate::{Error,Result};
extern "C" {
    fn LZ4_versionString()->*const c_char;
    fn LZ4_compressBound(n:c_int)->c_int;
    fn LZ4_sizeofState()->c_int;
    fn LZ4_compress_fast_extState(state:*mut c_void,src:*const c_char,dst:*mut c_char,n:c_int,cap:c_int,acc:c_int)->c_int;
    fn LZ4_decompress_safe(src:*const c_char,dst:*mut c_char,size:c_int,cap:c_int)->c_int;
    fn ZSTD_versionString()->*const c_char;
    fn ZSTD_compressBound(n:usize)->usize;
    fn ZSTD_createCCtx()->*mut c_void;
    fn ZSTD_freeCCtx(p:*mut c_void)->usize;
    fn ZSTD_createDCtx()->*mut c_void;
    fn ZSTD_freeDCtx(p:*mut c_void)->usize;
    fn ZSTD_compressCCtx(ctx:*mut c_void,dst:*mut c_void,cap:usize,src:*const c_void,n:usize,level:c_int)->usize;
    fn ZSTD_decompressDCtx(ctx:*mut c_void,dst:*mut c_void,cap:usize,src:*const c_void,n:usize)->usize;
    fn ZSTD_isError(code:usize)->u32;
    fn ZSTD_getErrorName(code:usize)->*const c_char;
}
#[derive(Clone,Copy,Debug)]
pub enum Codec { Lz4{acceleration:i32}, Zstd{level:i32} }
impl Codec {
    pub fn name(self)->String{match self {Self::Lz4{acceleration}=>format!("lz4-c-a{acceleration}"),Self::Zstd{level}=>format!("zstd-c-l{level}")}}
}
pub fn versions()->(String,String){
    // SAFETY: reference libraries return static, NUL-terminated version strings.
    unsafe{(CStr::from_ptr(LZ4_versionString()).to_string_lossy().into_owned(),CStr::from_ptr(ZSTD_versionString()).to_string_lossy().into_owned())}
}
fn zcheck(n:usize)->Result<usize>{
    unsafe{if ZSTD_isError(n)!=0{Err(Error::Backend(CStr::from_ptr(ZSTD_getErrorName(n)).to_string_lossy().into_owned()))}else{Ok(n)}}
}
struct Context{ptr:NonNull<c_void>,encode:bool}
impl Context{
    fn new(encode:bool)->Result<Self>{
        let p=unsafe{if encode{ZSTD_createCCtx()}else{ZSTD_createDCtx()}};
        Ok(Self{ptr:NonNull::new(p).ok_or(Error::SizeOverflow)?,encode})
    }
}
impl Drop for Context{fn drop(&mut self){unsafe{if self.encode{ZSTD_freeCCtx(self.ptr.as_ptr());}else{ZSTD_freeDCtx(self.ptr.as_ptr());}}}}
struct Chunk{bytes:Vec<u8>,raw:bool,raw_size:usize}
pub struct Encoded{codec:Codec,chunks:Vec<Chunk>,raw_size:usize,block_size:usize}
impl Encoded{
    pub fn encoded_size(&self)->usize{32+32*self.chunks.len()+self.chunks.iter().map(|c|c.bytes.len()).sum::<usize>()}
    pub fn compress(input:&[u8],codec:Codec,block_size:usize,threads:usize)->Result<Self>{
        crate::frame::check_block_size(block_size)?;
        if !(1..=256).contains(&threads){return Err(Error::InvalidOption("threads must be 1..256"));}
        if matches!(codec,Codec::Lz4{acceleration} if acceleration<1){return Err(Error::InvalidOption("LZ4 acceleration must be positive"));}
        let count=input.len().div_ceil(block_size);let workers=threads.min(count.max(1));
        let mut indexed=if workers==1{encode_group(input,codec,block_size,0,1)?}else{
            std::thread::scope(|scope|->Result<Vec<_>>{
                let mut handles=Vec::new();for w in 0..workers{handles.push(scope.spawn(move||encode_group(input,codec,block_size,w,workers)));}
                let mut all=Vec::new();for h in handles{all.extend(h.join().map_err(|_|Error::WorkerPanic)??);}Ok(all)
            })?
        };
        indexed.sort_unstable_by_key(|x|x.0);
        Ok(Self{codec,chunks:indexed.into_iter().map(|x|x.1).collect(),raw_size:input.len(),block_size})
    }
    pub fn decode_into(&self,out:&mut[u8],threads:usize)->Result<()>{
        if out.len()!=self.raw_size{return Err(Error::OutputSize{expected:self.raw_size,actual:out.len()});}
        if !(1..=256).contains(&threads){return Err(Error::InvalidOption("threads must be 1..256"));}
        if self.chunks.is_empty(){return Ok(());}
        let workers=threads.min(self.chunks.len());let per=self.chunks.len().div_ceil(workers);
        if workers==1{return decode_group(&self.chunks,out,self.codec);}
        std::thread::scope(|scope|{
            let mut handles=Vec::new();
            for (g,target) in out.chunks_mut(per*self.block_size).enumerate(){
                let start=g*per;let chunks=&self.chunks[start..(start+per).min(self.chunks.len())];let codec=self.codec;
                handles.push(scope.spawn(move||decode_group(chunks,target,codec)));
            }
            for h in handles{h.join().map_err(|_|Error::WorkerPanic)??;}Ok(())
        })
    }
}
fn encode_group(input:&[u8],codec:Codec,bs:usize,worker:usize,workers:usize)->Result<Vec<(usize,Chunk)>>{
    let ctx=if matches!(codec,Codec::Zstd{..}){Some(Context::new(true)?)}else{None};
    // LZ4 extState requires 8-byte alignment. Vec<u64>, not Vec<u8>.
    let mut state=if matches!(codec,Codec::Lz4{..}){
        let n=unsafe{LZ4_sizeofState()};if n<=0{return Err(Error::Backend("LZ4 state size".into()));}vec![0u64;(n as usize).div_ceil(8)]
    }else{Vec::new()};
    let mut all=Vec::new();
    for (i,src) in input.chunks(bs).enumerate().skip(worker).step_by(workers){
        let cap=unsafe{match codec{Codec::Lz4{..}=>LZ4_compressBound(src.len() as c_int) as usize,Codec::Zstd{..}=>ZSTD_compressBound(src.len())}};
        let mut dst=vec![0u8;cap];
        // SAFETY: bs <=1 MiB fits C int; all buffer capacities match their allocations;
        // external contexts are uniquely held by this worker.
        let n=unsafe{match codec{
            Codec::Lz4{acceleration}=>{
                let n=LZ4_compress_fast_extState(state.as_mut_ptr().cast(),src.as_ptr().cast(),dst.as_mut_ptr().cast(),src.len() as c_int,cap as c_int,acceleration);
                if n<=0{return Err(Error::Backend("LZ4 compression failed".into()));}n as usize
            }
            Codec::Zstd{level}=>zcheck(ZSTD_compressCCtx(ctx.as_ref().unwrap().ptr.as_ptr(),dst.as_mut_ptr().cast(),cap,src.as_ptr().cast(),src.len(),level))?,
        }};
        let raw=n>=src.len();if raw{dst.clear();dst.extend_from_slice(src);}else{dst.truncate(n);}
        all.push((i,Chunk{bytes:dst,raw,raw_size:src.len()}));
    }
    Ok(all)
}
fn decode_group(chunks:&[Chunk],out:&mut[u8],codec:Codec)->Result<()>{
    let ctx=if matches!(codec,Codec::Zstd{..}){Some(Context::new(false)?)}else{None};let mut p=0;
    for chunk in chunks{
        let dst=&mut out[p..p+chunk.raw_size];p+=chunk.raw_size;
        if chunk.raw{dst.copy_from_slice(&chunk.bytes);continue;}
        let n=unsafe{match codec{
            Codec::Lz4{..}=>{
                let n=LZ4_decompress_safe(chunk.bytes.as_ptr().cast(),dst.as_mut_ptr().cast(),chunk.bytes.len() as c_int,dst.len() as c_int);
                if n<0{return Err(Error::Backend("LZ4 decompression failed".into()));}n as usize
            }
            Codec::Zstd{..}=>zcheck(ZSTD_decompressDCtx(ctx.as_ref().unwrap().ptr.as_ptr(),dst.as_mut_ptr().cast(),dst.len(),chunk.bytes.as_ptr().cast(),chunk.bytes.len()))?,
        }};
        if n!=dst.len(){return Err(Error::Backend("baseline decoded size mismatch".into()));}
    }
    Ok(())
}
