const express = require('express');
const app = express();

app.use(express.urlencoded({ extended: false }))
app.use(express.json());
app.set("view engine", "ejs");

const postRoutes = require("./routes/posts");
app.use("/posts", postRoutes);

app.listen(4000, () => {
    console.log('Server is running on port 4000')
})
