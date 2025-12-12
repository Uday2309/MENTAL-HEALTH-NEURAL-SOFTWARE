import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'traffic-green': '#10b981',
        'traffic-amber': '#f59e0b',
        'traffic-red': '#ef4444',
      },
    },
  },
  plugins: [],
}
export default config


