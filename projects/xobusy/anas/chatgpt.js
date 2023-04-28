const express = require("express");
const app = express();
const bodyParser = require("body-parser");
const crypto = require("crypto");
const session = require("express-session");

// Parse incoming requests with JSON payloads
app.use(bodyParser.json());
app.use(express.static('public'))

// Use sessions for user management
app.use(session({
    secret: "my-secret-key",
    resave: false,
    saveUninitialized: true
}));

// Array to store user data
const users = [];

// Serve registration page
app.get("/register", (req, res) => {
    res.sendFile(__dirname + "/register.html");
});

app.get("/login", (req, res) => {
    res.sendFile(__dirname + "/login.html");
});

// Handle registration request
app.post("/register", (req, res) => {
    const { username, password } = req.body;

    // Generate a random salt
    const salt = crypto.randomBytes(16).toString("hex");

    // Hash the password with the salt using the SHA-256 algorithm
    const hash = crypto.createHash("sha256").update(password + salt).digest("hex");

    // Store the user data in the array
    users.push({ username, hash, salt });

    res.send("Registration successful");
});

// Handle login request
app.post("/login", (req, res) => {
    const { username, password } = req.body;

    // Find the user with the matching username
    const user = users.find(user => user.username === username);

    if (user) {
        // Hash the password with the user's salt and compare to the stored hash
        const hash = crypto.createHash("sha256").update(password + user.salt).digest("hex");

        if (hash === user.hash) {
            // Save the user ID to the session
            req.session.userId = username;

            // Redirect to the dashboard page if login is successful
            res.redirect("/dashboard");
        } else {
            res.send("Invalid username or password");
        }
    } else {
        res.send("Invalid username or password");
    }
});

// Example middleware to check if the user is authenticated
function isAuthenticated(req, res, next) {
    if (req.session.isAuthenticated) {
      return next();
    }
  
    // If the user is not authenticated, redirect to the login page
    res.redirect('/login');
  }
  
  // Example dashboard route that requires authentication
  app.get('/dashboard', isAuthenticated, function(req, res) {
    res.sendFile(path.join(__dirname, 'dashboard.html'));
  });

// Start the server
app.listen(3000, () => {
    console.log("Server started on port 3000");
});

