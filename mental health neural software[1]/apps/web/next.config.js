/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['@mindwatch/shared'],
  experimental: {
    serverActions: true,
  },
}

module.exports = nextConfig


