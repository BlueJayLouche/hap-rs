fn main() {
    let w = 1920usize; let h = 1080usize;
    let mut rgba = vec![0u8; w*h*4];
    for y in 0..h { for x in 0..w {
        let i = (y*w+x)*4;
        rgba[i] = (x%256) as u8; rgba[i+1] = (y%256) as u8;
        rgba[i+2] = ((x+y)%256) as u8; rgba[i+3] = 255;
    }}
    let mut out = vec![0u8; w*h];
    let t = std::time::Instant::now();
    hap_qt::bc7::compress_bc7_mode6(&rgba, w, h, &mut out);
    println!("1080p BC7 mode6 encode: {:?}", t.elapsed());
}
