#![no_main]
use libfuzzer_sys::fuzz_target;
fuzz_target!(|data:&[u8]|{
    let options=vxg::DecodeOptions{max_output:1<<20,..Default::default()};
    let safe=vxg::decompress(data,&options);
    if let Ok(frame)=vxg::Frame::parse(data,options.max_output){
        if let Ok(validated)=frame.into_validated(){
            let mut out=vec![0;validated.raw_size()];
            let fast=validated.decode_into(&mut out,&options).map(|_|out);
            assert_eq!(fast,safe);
        }
    }
});
