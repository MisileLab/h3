
export interface CodeTemplate {
  name: string;
  description: string;
  content: string;
}

export const templates: CodeTemplate[] = [
  {
    name: 'Basic HTML Page',
    description: 'A minimal HTML5 page structure.',
    content: `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>New Page</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Hello from Template!</h1>
    <script src="script.js"></script>
</body>
</html>`,
  },
  {
    name: 'Styled Button',
    description: 'A simple, styled button in CSS.',
    content: `/* style.css */
.my-button {
  background-color: #4CAF50; /* Green */
  border: none;
  color: white;
  padding: 15px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 2px;
  cursor: pointer;
  border-radius: 8px;
}`,
  },
  {
    name: 'JavaScript Alert',
    description: 'A simple JavaScript alert function.',
    content: `// script.js
function showAlert() {
  alert('Hello from JavaScript!');
}

showAlert();`,
  },
];
