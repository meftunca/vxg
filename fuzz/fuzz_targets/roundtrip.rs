#![no_main]
use libfuzzer_sys::fuzz_target;
fuzz_target!(|data:&[u8]|{
    if data.len()>1<<20{return;}
    let options=vxg::Options{block_size:4096,..Default::default()};
    let (encoded,work)=vxg::compress_with_stats(data,&options).unwrap();
    assert!(encoded.len()<=vxg::max_compressed_size(data.len(),options.block_size).unwrap());
    assert!(work.probes<=data.len() as u64);assert!(work.matched_bytes<=data.len() as u64);
    assert_eq!(vxg::decompress(&encoded,&Default::default()).unwrap(),data);
    let v3=vxg::Options{block_size:4096,..Default::default()};
    let encoded=vxg::compress(data,&v3).unwrap();
    assert!(encoded.len()<=vxg::max_compressed_size(data.len(),v3.block_size).unwrap());
    assert_eq!(vxg::decompress(&encoded,&Default::default()).unwrap(),data);
});
