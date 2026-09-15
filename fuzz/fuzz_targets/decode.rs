#![no_main]
use libfuzzer_sys::fuzz_target;
fuzz_target!(|data:&[u8]|{
    let options=vxg::DecodeOptions{max_output:1<<20,..Default::default()};
    let _=vxg::decompress(data,&options);
});
