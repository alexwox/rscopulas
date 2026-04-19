import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "media.brand.dev",
      },
      {
        protocol: "https",
        hostname: "pub-ca706ca75c8a4972b721945607f0ff01.r2.dev",
      },
    ],
  },
};

export default nextConfig;
