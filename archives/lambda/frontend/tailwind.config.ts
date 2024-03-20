/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
	"./src/**/*.{html,js,jsx,md,mdx,ts,tsx}"
  ],
  presets: [require("./ui.preset.js")],
  theme: {
    screens: {
      flip: '300px',
      smp: '400px',
      sm: '640px',
      md: '768px',
      lg: '1024px',
      xl: '1280px',
      '2xl': '1536px'
    }
  }
}
