/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  daisyui: {
    themes: ["light", "dark", "retro", "lemonade"],
  },
  plugins: [
    require("daisyui"),
    require('@tailwindcss/typography'),
  ],
}