import { render } from "solid-js/web";
import { Router, Route, Routes } from "@solidjs/router";
import App from './App';
import "./global.css";

render(
	() => (
		<Router>
			<Routes>
				<Route path="/" component={App} />
			</Routes>
		</Router>
	),
	document.getElementById("root") as HTMLElement
);