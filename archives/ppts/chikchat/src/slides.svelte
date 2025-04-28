<script lang="ts">
	import { Code, Media, Notes, Presentation, Slide, Vertical } from '@components'
</script>

<Presentation>
	<Slide animate>
		<p class="font-bold">Chikchat</p>
		<p class="mt-5">간단한 채팅과 일정 앱</p>
	</Slide>
	<Vertical>
		<Slide>
			<p class="font-bold mb-5">목차</p>
			<p class="fragment highlight-green mb-3">1. 소개/만든 이유</p>
			<p class="mb-3">2. 시연</p>
			<p class="mb-3">3. 코드 설명</p>
			<p>4. 느낀 점</p>
		</Slide>
		<Slide>
			<p class="font-bold mb-5 mt-0">소개</p>
			<div class="text-left">
				<!-- svelte-ignore a11y-img-redundant-alt -->
				<Media src="https://avatars.githubusercontent.com/u/72301973?v=4" alt="minyee2913's image" class="w-1/4 inline" type="img" />
				<div class="ml-5 inline">
					<p class="font-bold inline">이강민</p>
					<p class="inline">- 프론트엔드, ppt</p>
				</div>
				<Media src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Tux.svg/1727px-Tux.svg.png" alt="tux" class="w-1/4 inline" type="img" />
				<div class="ml-5 inline">
					<p class="font-bold inline">설지원</p>
					<p class="inline">- 백엔드, 프론트엔드, ppt</p>
				</div>
			</div>
		</Slide>
		<Slide>
			<p class="font-bold mb-5">만든 이유: 내가 쓸려고</p>
			<p>- 이강민</p>
		</Slide>
	</Vertical>
	<Vertical>
		<Slide>
			<p class="font-bold mb-5">목차</p>
			<p class="mb-3">1. 소개/만든 이유</p>
			<p class="fragment highlight-green mb-3">2. 시연</p>
			<p class="mb-3">3. 코드 설명</p>
			<p>4. 느낀 점</p>
		</Slide>
	</Vertical>
	<Vertical>
		<Slide>
			<p class="font-bold mb-5">목차</p>
			<p class="mb-3">1. 소개/만든 이유</p>
			<p class="mb-3">2. 시연</p>
			<p class="fragment highlight-green mb-3">3. 코드 설명</p>
			<p>4. 느낀 점</p>
		</Slide>
		<Slide>
			<p class="font-bold mb-10">프론트엔드 기술</p>
			<div style="inline">
				<Media src="https://blog.kakaocdn.net/dn/cUev93/btq1z5ugvxh/UIrrExYF9dTLthqSQ5mBak/img.png" type="img" class="w-1/4 inline"></Media>
				<p class="mt-10">Electron</p>
			</div>
		</Slide>
		<Slide>
			<p class="font-bold mb-10">백엔드 기술</p>
			<div class="flex justify-between">
				<div class="w-1/2 flex flex-col items-center">
					<Media type="img" src="./public/deno.png" class="w-1/4 mb-5" />
					<p>Deno</p>
				</div>
				<div class="w-1/2 flex flex-col items-center">
					<Media type="img" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRcAB-vVMYP2er005ithWJadA3WsALiempXUpsErytUtQNnVwOJee_X7oJCpbmnVANJYpg&usqp=CAU" class="w-1/4 mb-10" />
					<p>EdgeDB</p>
				</div>
			</div>
		</Slide>
		<Slide>
			프론트엔드 설명 알아서 강민이 ㄱㄱ
		</Slide>
		<Slide>
			<Code lang="ts" lines="1-2|10-25|18-24|28-44|29-36|39-44|53-67|60-66|68|72-87|77-81|96-102|105-120|113-119|124-131|134-143">
				{`
				import { Application, Router } from "https://deno.land/x/oak/mod.ts";
				import * as edgedb from "https://deno.land/x/edgedb/mod.ts";

				const app = new Application();
				const router = new Router();
				const db = edgedb.createClient();
				let message = {};

				// 실시간 채팅을 위한 웹소켓
				router.get("/ws", (ctx) => {
				if (!ctx.isUpgradable) {
					ctx.response.status = 400;
					return;
				}
				const ws = ctx.upgrade();
				let _original = {}
				// 채팅 데이터가 바뀔 경우 바뀐 데이터 전송
				setInterval(()=>{
					if (message == _original) {return;}
					if (ws.readyState != 1) {return;}
					console.log(message);
					_original = message;
					ws.send(JSON.stringify(message));
				}, 1000);
				})

				// 유저 정보 전송
				router.get("/users", async (ctx) => {
				const users = await db.queryJSON(\`
				select Account {
					username,
					accid,
					description,
					image,
					manage
				}\`);
				console.log(users);
				console.log("a");
				const jusers = JSON.parse(users);
				for (const i of jusers) {
					if (i.manage.toString() !== "") {
					console.log(typeof i.manage);
					console.log(i.manage);
					i.manage = JSON.parse(i.manage)
					}
				}
				console.log(jusers);
				ctx.response.status = 200;
				ctx.response.body = jusers;
				})

				// 회원가입 요청
				router.post("/register", async (ctx) => {
				const {headers} = ctx.request;
				const username = headers.get("username");
				const accid = headers.get("accid");
				const password = headers.get("password");
				const description = headers.get("description");
				// 만약 accid가 이미 존재한다면 409 Conflict
				if (await db.querySingle(\`select Account 
				{ username, accid, description, image, manage } 
				filter .accid = <str>$accid\`, {accid}) != null) {
					ctx.response.status = 409;
					ctx.response.body = "accid already exists";
					return;
				}
				ctx.response.status = 201;
				await db.querySingle("insert Account { username := <str>$username, accid := <str>$accid, password := <str>$password, description := <str>$description}", {username, accid, password, description});
				})

				// 로그인 요청
				router.get("/login", async (ctx) => {
				const {headers} = ctx.request;
				const accid = headers.get("accid");
				const password = headers.get("password");
				// accid와 password가 일치하는 유저가 없다면 401 Unauthorized
				const user = await db.querySingle("select Account { username, accid, description, image, manage } filter .accid = <str>$accid and .password = <str>$password", {accid, password});
				if (user == null) {
					ctx.response.status = 401;
					return;
				}
				if (user.manage.toString() != "") {
					user.manage = JSON.parse(user.manage)
				}
				ctx.response.status = 200;
				ctx.response.body = user;
				})

				// 메시지 리스트 요청
				router.get("/msgs", async (ctx) => {
				const msg = await db.queryJSON("select Message {from, to, text, time}");
				ctx.response.body = msg;
				})

				// 메시지 요청
				router.get("/msg", async (ctx) => {
				const {headers} = ctx.request;
				const from = headers.get("from");
				const msg = await db.queryJSON("select Message {from, to, text, time} filter .from = <str>$from", {from});
				ctx.response.status = 200;
				ctx.response.body = msg;
				})

				// 메시지 전송 요청
				router.post("/msg", async (ctx) => {
				const headerlist = ctx.request.headers;
				const from = headerlist.get("from");
				const to = headerlist.get("to");
				const content = headerlist.get("content");
				const time = Date.now();
				const json = await db.querySingleJSON("insert Message {from := <str>$from, to := <str>$to, text := <str>$content, time := <int64>$time}", {from, to, content, time});
				// 메시지 변경 전송
				message = {
					"id": JSON.parse(json)["id"],
					"from": from,
					"content": content,
					"time": time,
					"to": to
				};
				ctx.response.body = "ok";
				ctx.response.status = 201;
				})

				// 설명 변경
				router.post("/update_desc", async (ctx) => {
				const {headers} = ctx.request;
				const accid = headers.get("accid");
				const description = headers.get("description");
				await db.querySingle("update Account filter .accid = <str>$accid set {description := <str>$description}", {accid, description});
				ctx.response.status = 200;
				});

				// 일정 변경
				router.post("/update_manage", async (ctx) => {
				const {headers} = ctx.request;
				const accid = headers.get("accid");
				const manage = headers.get("manage");
				if (manage === null) {ctx.response.status = 400; return;}
				// 다 지우고 새로 데이터 연동
				console.log("a");
				await db.querySingle("update Account filter .accid = <str>$accid set {manage := <json>$manage}", {accid, manage});
				ctx.response.status = 200;
				})

				app.use(router.allowedMethods());
				app.use(router.routes());

				await app.listen({ port: 21000 });`}
			</Code>
		</Slide>
	</Vertical>
	<Vertical>
		<Slide>
			<p class="font-bold mb-5">목차</p>
			<p class="mb-3">1. 소개/만든 이유</p>
			<p class="mb-3">2. 시연</p>
			<p class="mb-3">3. 코드 설명</p>
			<p class="fragment highlight-green">4. 느낀 점</p>
		</Slide>
		<Slide>
			<p class="mb-5">JS로 앱을 만드는 게 재밌었다 - 이강민</p>
			<p class="mb-5">JavaScript는 웹에서만 써야 한다 - 설지원</p>
		</Slide>
	</Vertical>
</Presentation>

<style>
	@font-face {
		font-family: 'NanumSquareNeo-Variable';
		src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_11-01@1.0/NanumSquareNeo-Variable.woff2') format('woff2');
		font-weight: normal;
		font-style: normal;
	}
	* {
		font-family: 'NanumSquareNeo-Variable';
	}

</style>
