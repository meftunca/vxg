class Vxg < Formula
  desc "Adaptive block-independent lossless codec & compressed-domain SQL/Arrow engine"
  homepage "https://meftunca.github.io/vxg/"
  url "https://github.com/meftunca/vxg/archive/refs/tags/v0.6.1.tar.gz"
  version "0.6.1"
  license "MIT"
  head "https://github.com/meftunca/vxg.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "build", "--release", "--bin", "vxg"
    system "cargo", "build", "--release", "--lib"
    bin.install "target/release/vxg"
    lib.install "target/release/libvxg.a"
    include.install "include/vxg.h"
  end

  test do
    assert_match "vxg", shell_output("#{bin}/vxg --help 2>&1")
  end
end
