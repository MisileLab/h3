const express = require("express");
const session = require("express-session");
const bodyParser = require('body-parser');
const crypto = require('crypto');

const server = express();
const users = {
}

function encryptString(originalString) {
    // Generate a random salt value
    const salt = crypto.randomBytes(16).toString('hex');

    // Hash the string using SHA-256 algorithm with the salt
    const hash = crypto.createHmac('sha256', salt)
                       .update(originalString)
                       .digest('hex');

    // Return the salt and encrypted text as an object
    return { salt, hash };
}

function compareString(originalString, salt, encryptedText) {
    // Hash the original string using SHA-256 algorithm with the salt
    const hash = crypto.createHmac('sha256', salt)
                       .update(originalString)
                       .digest('hex');

    // Compare the hashed original string with the encrypted text
    return hash === encryptedText;
}

server.use(session({
    key: 'sessionid',
    secret: 'mysecret',
    cookie: { expires: false },
    resave: false,
    saveUninitialized: true
}))
server.use(bodyParser.urlencoded({extended: true}))
server.use(bodyParser.json())

server.get("/setSession", (req, res) => {
    req.session.test = 'abc'
    res.redirect('/')
})

server.all("/register", (req, res) => {
    if (!['GET', 'POST'].includes(req.method)) {
        return res.status(405).send("this method not allowed")
    }

    let tent = `
        <form method=post>
            <div>
                <label>사용자 ID: <label><br>
                <input type=text name=username />
            </div>
            <div>
                <label>비밀번호: <label><br>
                <input type=password name=password />
            </div>
            <div>
                <button class=btn type=submit>register</button>
            </div>
        </form>
    `

    console.log(req.method)

    if (req.method == 'POST') {
        let estring = encryptString(req.body.password);
        users[req.body.username] = {
            name: req.body.username,
            password: estring['hash'],
            salt: estring['salt']
        }
        console.log(users)
    }

    res.send(tent);
})

server.get('/', (req, res) => {
    console.log('[ 세션 쿠키 정보 ]');
    console.log(req.session.cookie);
    
    let content = `
        <p>현재 세션 정보: </p>
        <ul>
    `;
    for(let session in req.session)
        content += `<li>${session}: ${req.session[session]}</li>`;
    content += '</ul>';

    if (req.session.username) {
        content += `<p>login to ${req.session.username}</p>`
    } else {
        content += '<p>no login? go to <a href=/login>/login</a></p>'
    }
    
    res.send(content);
});

server.all("/login", (req, res) => {
    if (!['GET', 'POST'].includes(req.method)) {
        return res.status(405).send("this method not allowed")
    }

    let tent = `
        <form method=post>
            <div>
                <label>사용자 ID: <label><br>
                <input type=text name=username />
            </div>
            <div>
                <label>비밀번호: <label><br>
                <input type=password name=password />
            </div>
            <div>
                <button class=btn type=submit>login</button>
            </div>
        </form>
    `

    if (req.method == 'POST') {
        console.log(users)
        if (compareString(req.body.password, users[req.body.username].salt, users[req.body.username].password)) {
            req.session.username = req.body.username;
            res.redirect('/')
        } else {
            tent = `
                you can't login because no auth
            ` + tent;
            return res.status(400).send(tent)
        }
    }

    res.send(tent);
})

server.listen(8000);
