<script lang="ts">
	import { Code, Media, Notes, Presentation, Slide, Vertical } from '@components'
</script>

<Presentation>
	<Slide animate>
		<p class="font-bold">Qubit</p>
		<p class="mt-5">보안 취약점 스캔 GUI</p>
		<Notes>This is start!</Notes>
	</Slide>
	<Vertical>
		<Slide>
			<p class="font-bold mb-5">목차</p>
			<p class="fragment custom highlight-cyan mb-3">1. 만든 이유</p>
			<p class="mb-3">2. 시연</p>
			<p class="mb-3">3. 코드 설명</p>
			<p>4. 개선할 점</p>
		</Slide>
		<Slide>
			<Notes>보안 취약점이 일어나는 이유 -> 사람의 문제가 큼</Notes>
			<p class="font-bold mb-10">보안 취약점이 일어나는 이유</p>
			<center><img src="./images/ibm-and-verizon.png" alt="https://www.varonis.com/blog/cybersecurity-statistics"></center>
			<p class="mt-5">74%의 보안 취약점은 사람의 문제로 발생됨</p>
			<p class="mt-5">보안 취약점을 알아차리는데 걸리는 시간은 207일</p>
		</Slide>
		<Slide>
			<Notes>위험한 걸 물어보고 -> 자신의 생각으로 자연스럽게 전환</Notes>
			<p class="font-bold mb-10">사람이 실수하는 것 중에서 가장 위험한 건 뭘까?</p>
			<div class="fragment">
				<center><img src="./images/twitter-security.png" alt="https://www.boannews.com/media/view.asp?idx=108847"></center>
				<p class="mt-10">API 키 유출</p>
			</div>
		</Slide>
		<Slide>
			<p class="font-bold mb-10">API 키 스캔 툴이 있음, 그러나</p>
			<center class="content-end">
				<img src="./images/gitleaks.png" alt="https://github.com/gitleaks/gitleaks">
				<p class="mt-5">쉽게 사용할 수 없음</p>
			</center>
		</Slide>
		<Slide>
			<p class="font-bold">GUI로 쉽게 볼 수는 없을까?</p>
		</Slide>
	</Vertical>
	<Vertical>
		<Slide>
			<p class="font-bold mb-5">목차</p>
			<p class="mb-3">1. 만든 이유</p>
			<p class="mb-3 fragment custom highlight-cyan">2. 시연</p>
			<p class="mb-3">3. 코드 설명</p>
			<p>4. 개선할 점</p>
		</Slide>
		<Slide>
			<Notes>영상은 그냥 보여주고 다음 화면들에서 상세 설명</Notes>
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
			<p class="mb-3 fragment custom highlight-cyan">3. 코드 설명</p>
			<p>4. 개선할 점</p>
		</Slide>
		<Slide>
			<p class="font-bold mb-10">사용한 기술</p>
			<div class="flex justify-between">
				<div class="w-1/2 flex flex-col items-center">
					<Media type="img" src="/images/nextjs.png" class="w-1/4 mb-5" />
					<p>Next.JS</p>
				</div>
				<div class="w-1/2 flex flex-col items-center">
					<Media type="img" src="/images/tauri.svg" class="w-3/4 mb-6" />
					<p>Tauri</p>
				</div>
			</div>
		</Slide>
		<Slide>
			<Notes>
				이 코드는 줄인 코드라고 말해야함<br>
				(너무 길기 때문에 줄인 것)<br>
				간략한 코드 설명<br>
				1. tauri command 초기화<br>
				2. CLI 출력 파싱<br>
				3. 콘피그 파일 저장
			</Notes>
			<p class="font-bold mb-10">스캔 핵심 코드</p>
			<Code lang="tsx" class="w-screen ml-0" lines="1|4-13|15-16">
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
			<Notes>
				간략한 코드 설명<br>
				1. 보안 취약점 선택<br>
				2. 보안 취약점 보여주는 버튼 핸들링<br>
				3. 보안 취약점 결과 보여주는 코드
			</Notes>
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
			<p class="fragment custom highlight-cyan">4. 개선할 점</p>
		</Slide>
		<Slide>
			<p class="font-bold">개선할 점</p>
			<p class="mt-5">1. 코드가 너무 복잡함</p>
			<p class="mt-5">2. 자동 셋업 기능 추가</p>
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
