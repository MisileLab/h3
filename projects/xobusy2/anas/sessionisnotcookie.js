const express = require("express");
const session = require("express-session");
const bodyParser = require('body-parser');

// this is not vaild code!!!!! I have a no time so I can't fix it

const server = express();
const users = {
    "AnA": "thispasswordissecure"
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

    if (req.method == 'post') {
        if (users[req.body.username] == req.body.password) {
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
