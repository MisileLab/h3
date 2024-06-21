/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {
			colors: {
			  highlight: '#7d92e3',
			  white: '#e4e4e4',
			  grayText: '#aaaaaa',
			  gray: '#838383',
			  background: '#181818',
			  backgroundDarker: '#141414',
			  backgroundLay2: '#1f2024',
			  backgroundLay3: '#212328',
			  backgroundLay4: '#1a1c1f',
			  border: '#3c3c3c'
			}
		},
	},
	plugins: []
}
