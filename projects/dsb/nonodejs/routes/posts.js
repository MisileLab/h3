const express = require('express');
const router = express.Router();
const Post = require('../models/post');
const Reply = require('../models/reply');

router.get('/', async (req, res) => {
    res.render("home");
});

module.exports = router;
