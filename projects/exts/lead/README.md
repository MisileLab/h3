# Google Form Lead Manager

This is a web application built with [Astro](https://astro.build/) to fetch, display, and export leads from a Google Form. It uses the Google Forms API to retrieve data.

## ✨ Core Technologies

*   **Frontend:** [Astro](https://astro.build/)
*   **Styling:** [Tailwind CSS](https://tailwindcss.com)
*   **Client-side Database:** [Dexie.js](https://dexie.org/) (IndexedDB) - *Note: Initial setup is in place, but it is not yet integrated into the main application.*
*   **Excel Export:** [SheetJS/xlsx](https://sheetjs.com/)
*   **Google APIs:** Google API Client for JavaScript

## 🚀 Features

*   Authorize and connect to the Google Forms API using OAuth 2.0.
*   Fetch all responses from a specified Google Form.
*   Display the responses in a table.
*   Export the captured leads to an `.xlsx` Excel file.
*   A fun confetti button component!

## 🔧 Setup & Configuration

To run this application, you will need credentials for the Google Forms API.

1.  **Google Form ID:** The ID of the Google Form you want to fetch responses from.
2.  **Google Web App Client ID:** You'll need to create an OAuth 2.0 Client ID in the Google Cloud Console for a web application.

These values need to be entered into the input fields on the main page.

## 🧞 Commands

All commands are run from the root of the project, from a terminal:

| Command                | Action                                           |
| :--------------------- | :----------------------------------------------- |
| `pnpm install`         | Installs dependencies                            |
| `pnpm dev`             | Starts local dev server at `localhost:4321`      |
| `pnpm build`           | Build your production site to `./dist/`          |
| `pnpm preview`         | Preview your build locally, before deploying     |
