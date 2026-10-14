# typed: false
# frozen_string_literal: true

class Ramllm < Formula
  desc "Run bigger models on smaller machines — RAM-adaptive, GPU-aware LLM inference manager"
  homepage "https://github.com/Mayank3613/RAMLLM"
  version "0.1.0"
  license "Apache-2.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/Mayank3613/RAMLLM/releases/download/v#{version}/ramllm-darwin-arm64"
      # sha256 is generated in release workflow SHA256SUMS.txt
      sha256 "REPLACE_WITH_DARWIN_ARM64_SHA256"

      def install
        bin.install "ramllm-darwin-arm64" => "ramllm"
      end
    else
      url "https://github.com/Mayank3613/RAMLLM/releases/download/v#{version}/ramllm-darwin-x86_64"
      sha256 "REPLACE_WITH_DARWIN_X86_64_SHA256"

      def install
        bin.install "ramllm-darwin-x86_64" => "ramllm"
      end
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/Mayank3613/RAMLLM/releases/download/v#{version}/ramllm-linux-x86_64"
      sha256 "REPLACE_WITH_LINUX_X86_64_SHA256"

      def install
        bin.install "ramllm-linux-x86_64" => "ramllm"
      end
    end
  end

  test do
    assert_match "RAMLLM", shell_output("#{bin}/ramllm --help")
  end
end
