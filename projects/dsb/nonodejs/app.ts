import express from 'npm:express';
const app = express();

app.use(express.urlencoded({ extended: false }))
app.use(express.json());

app.listen(4000, () => {
    console.log('Server is running on port 4000')
})
