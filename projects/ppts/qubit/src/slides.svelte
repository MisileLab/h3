<script lang="ts">
	import { Code, Media, Notes, Presentation, Slide, Vertical } from '@components'
</script>

<Presentation>
	<Slide animate>
		<p class="font-bold">Qubit</p>
		<p class="mt-5">보안 취약점 스캔 GUI</p>
		<Notes>This is just sample note</Notes>
	</Slide>
	<Vertical>
		<Slide>
			<p class="font-bold mb-5">목차</p>
			<p class="fragment highlight-green mb-3">1. 만든 이유</p>
			<p class="mb-3">2. 시연</p>
			<p class="mb-3">3. 코드 설명</p>
			<p>4. 개선할 점</p>
		</Slide>
		<Slide>
			<p class="font-bold mb-10">보안 취약점이 일어나는 이유</p>
			<center><img src="./images/ibm-and-verizon.png" alt="https://www.varonis.com/blog/cybersecurity-statistics"></center>
			<p class="mt-5">74%의 보안 취약점은 사람의 문제로 발생됨</p>
			<p class="mt-5">보안 취약점을 알아차리는데 걸리는 시간은 207일</p>
		</Slide>
		<Slide>
			<p class="font-bold mb-10">사람이 실수하는 것 중에서 가장 위험한 건 뭘까?</p>
			<center><img src="./images/twitter-security.png" alt="https://www.boannews.com/media/view.asp?idx=108847" class="fragment"></center>
			<p class="fragment mt-10">API 키 유출</p>
		</Slide>
		<Slide>
			<p class="font-bold mb-10">API 키 스캔 툴이 있음, 그러나</p>
			<center class="content-end">
				<img src="./images/gitleaks.png" alt="https://github.com/gitleaks/gitleaks">
				<p class="mt-5">쉽게 사용할 수 없음</p>
			</center>
		</Slide>
		<Slide>
			<p class="font-bold">GUI로 쉽게 코드 스캔할 수는 없을까?</p>
		</Slide>
	</Vertical>
	<Vertical>
		<Slide>
			<p class="font-bold mb-5">목차</p>
			<p class="mb-3">1. 만든 이유</p>
			<p class="mb-3 fragment highlight-green">2. 시연</p>
			<p class="mb-3">3. 코드 설명</p>
			<p>4. 개선할 점</p>
		</Slide>
		<Slide>
			<center><Media src="./video.mp4" autoplay={true} type="video"></Media></center>
		</Slide>
		<Slide>
			<p class="font-bold mb-20">메인 화면</p>
			<center><img src="./images/app/main.png" alt="just main page"></center>
			<p class="mt-10">스캔 횟수, 보안 취약점 찾은 횟수</p>
		</Slide>
		<Slide>
			<p class="font-bold mb-20">스캔 화면</p>
			<center><img src="./images/app/newScan.png" alt="just newScan page"></center>
			<p class="mt-10">스캔 경로 지정과 스캔 시작 버튼</p>
		</Slide>
		<Slide>
			<p class="font-bold mb-20">결과 화면</p>
			<center><img src="./images/app/scanres.png" alt="just scan without modal"></center>
			<p class="mt-10">스캔 정보와 찾은 보안 취약점 목록</p>
		</Slide>
		<Slide animate>
			<p class="font-bold mb-20">결과 화면</p>
			<center><img src="./images/app/scanresmodal.png" alt="just modal"></center>
			<p class="mt-10">어디서 API 키가 유출되었는지 알려줌</p>
		</Slide>
	</Vertical>
	<Vertical>
		<Slide>
			<p class="font-bold mb-5">목차</p>
			<p class="mb-3">1. 만든 이유</p>
			<p class="mb-3">2. 시연</p>
			<p class="mb-3 fragment highlight-green">3. 코드 설명</p>
			<p>4. 개선할 점</p>
		</Slide>
		<Slide>
			<p class="font-bold mb-10">사용한 기술</p>
			<p class="mb-5">Next.JS - 프론트엔드</p>
			<p>Tauri - 앱 번들링</p>
		</Slide>
		<Slide>
			<Notes>
				이 코드는 줄인 코드라고 말해야함
				(너무 길기 때문에 줄인 것)
			</Notes>
			<p class="font-bold mb-10">스캔 핵심 코드</p>
			<Code lang="tsx" class="w-screen ml-0" lines="1|2-13|4|5-12|15-16|20">
				{`
				let cmd2 = new Command('run-snyk');
				cmd2.on("close", async (_) => {
				if (!await exists("report_snyk.json", {dir: BaseDirectory.AppCache})) {return;}
				const _data2 = JSON.parse(await readTextFile("report_snyk.json"))
				for (const i of _data2.runs[0].results) {
					data.resnum.vulfound++;
					data.scans[length].leaks.push({
						name: 'found vulnerability in file',
						description: i.message.text,
						Line: i.locations[0].physicalLocation.region.startLine,
						Column: i.locations[0].physicalLocation.region.startColumn
					})
				}
				console.log(data, _data2);
				await writeTextFile("data.json", JSON.stringify(data));
				await removeFile("report_snyk.json", {dir: BaseDirectory.AppCache});
				})
				cmd2.stdout.on("data", a=>console.log(a))
				cmd2.stderr.on("data",e=>console.warn(e))
				await cmd2.spawn()
					`}
			</Code>
		</Slide>
		<Slide>
			<p class="font-bold mb-10">결과 확인 코드</p>
			<Code lang="tsx" class="w-screen ml-0" lines="8-10|21-24|28-32">
				{`
				<div className="bg-white dark:bg-gray-800 rounded-md shadow-sm p-6 mb-4">
				<div className="flex justify-between items-center mb-2">
					<span className="text-lg font-medium">Scan # {num}</span>
					<span className="text-sm text-gray-500 dark:text-gray-400">10/11/2023</span>
				</div>
				<div className="text-sm text-gray-600 dark:text-gray-400">
					<p>Path: {path}</p>
					<Select onValueChange={(s)=>{
						setOpenState({"index": Number(s), "opened": false})
					}}>
						<SelectTrigger>
							<SelectValue placeholder="Select a leak" />
						</SelectTrigger>
						<SelectContent>
							<SelectGroup className="bg-white">
								<SelectLabel>Leaks</SelectLabel>
								{items()}
							</SelectGroup>
						</SelectContent>
					</Select>
					<Button className="mt-2" variant="outline" onClick={()=>{setOpenState({
						"index": openState["index"],
						"opened": leaks.length != 0
					});}}>
						Show Result
					</Button>
				</div>
				{openState["opened"] ? <ScanRes leak={leaks[openState["index"]]} callback={
					()=>{setOpenState({
					"index": openState["index"],
					opened: false
				})}}/> : null}
			</div>
			`}
			</Code>
		</Slide>
	</Vertical>
	<Vertical>
		<Slide>
			<p class="font-bold mb-5">목차</p>
			<p class="mb-3">1. 만든 이유</p>
			<p class="mb-3">2. 시연</p>
			<p class="mb-3">3. 코드 설명</p>
			<p class="fragment highlight-green">4. 개선할 점</p>
		</Slide>
		<Slide>
			<p class="font-bold">개선할 점</p>
			<p class="mt-5">1. 코드 리팩토링 (현재 코드는 재사용이 어려움)</p>
			<p class="mt-5">2. 자동 셋업 (현재 앱은 툴을 직접 설치해야함)</p>
		</Slide>
		<Slide>
			<p class="font-bold text-4xl">사이버 공격은 모든 회사에 나올 수 있는 가장 큰 위협이다.</p>
			<p class="font-light mt-5 text-3xl">- 버지니아 로메티(전 IBM CEO)</p>
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
