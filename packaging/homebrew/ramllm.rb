# typed: false
# frozen_string_literal: true

class Ramllm < Formula
  desc "Run bigger models on smaller machines — RAM-adaptive, GPU-aware LLM inference manager"
  homepage "https://github.com/Mayank3613/RAMLLM"
  version "0.1.0"
  license "Apache-2.0"

  on_macos do
    url "https://github.com/Mayank3613/RAMLLM/releases/download/v#{version}/ramllm-darwin-arm64"
    sha256 "b15962071ef3256a4b88403171eb864129a30cf5d48a92e54acf077c5e3f0fc3"

    def install
      bin.install "ramllm-darwin-arm64" => "ramllm"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/Mayank3613/RAMLLM/releases/download/v#{version}/ramllm-linux-x86_64"
      sha256 "4a5685d7c6d8b12c27f49b05bb2e578a461d983af17bc5e1f6c60847e56eff03"

      def install
        bin.install "ramllm-linux-x86_64" => "ramllm"
      end
    end
  end

  test do
    assert_match "RAMLLM", shell_output("#{bin}/ramllm --help")
  end
end
