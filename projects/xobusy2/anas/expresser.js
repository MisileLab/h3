const express = require('express');
const cookieParser = require('cookie-parser');
const bodyParser = require('body-parser');
// file deepcode ignore PT: <please specify a reason of ignoring this></please>, file deepcode ignore UseCsurfForExpress: <please specify a reason of ignoring this>
const server = express();
server.use(cookieParser());
server.use(bodyParser.urlencoded(
    {extended: true}
));
server.use(bodyParser.json({
    limit: '50mb'
}));

server.get('/', (req, res) => {
    res.send(`
        <FORM method=post>
            <div>
                <label>cookie name: </label>
                <input type=text name=cookie-name />
            </div>

            <div>
                <label>value: </label>
                <input type=text name=cookie-value />
            </div>

            <div>
                <label>http only: </label>
                <input type=checkbox name=httpOnly value=false />
            </div>

            <div>
                <button type=submit>Save</button>
            </div>
        </FORM>
    `)
})

server.post('/', (req, res) => {
    res.cookie(req.body['cookie-name'], req.body['cookie-value'], {
        secure: false,
        maxAge: 86400000,
        httpOnly: !!req.body['httpOnly']
    });
    res.redirect('/bitcoinethereumespeciallydogecoin');
})

server.get('/verysecretcoin', (req, res) => {
    res.cookie('dogecoin-price', '10000%', {
        secure: false,
        httpOnly: false,
        maxAge: 86400000
    })
    res.send("cookie saved");
});

server.get('/bitcoinethereumespeciallydogecoin', (req, res) => {
    let content = `
    <h1>verysecretlist</h1>
    <ul>
    `;
    for (let cookie in req.cookies) {
        content += `
        <li>${cookie}: ${req.cookies[cookie]}</li>
        `;
    }
    content += "</ul>";
    res.setHeader('Content-Type', 'application/json')
    res.send(content);
})

server.listen(8000);
