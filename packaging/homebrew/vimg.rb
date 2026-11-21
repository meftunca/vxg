class Vimg < Formula
  desc "HTTP/HTTPS Range-native hierarchical wavelet image & animation codec (8/10/12/16-bit HDR, SIMD)"
  homepage "https://meftunca.github.io/vxg/VIMG"
  url "https://github.com/meftunca/vxg/archive/refs/tags/vimg-v0.4.0.tar.gz"
  version "0.4.0"
  license "MIT"
  head "https://github.com/meftunca/vxg.git", branch: "main"

  depends_on "rust" => :build

  def install
    cd "crates/vimg" do
      system "cargo", "build", "--release", "--features", "io", "--bin", "vimg"
      system "cargo", "build", "--release", "--lib"
      bin.install "target/release/vimg"
      lib.install "target/release/libvimg.a"
      include.install "include/vimg.h"
    end
  end

  test do
    assert_match "vimg #{version}", shell_output("#{bin}/vimg --help 2>&1")
  end
end
