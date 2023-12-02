<script lang="ts">
	import { Notes, Presentation, Slide, Vertical } from '@components'
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
			<p class="fragment highlight-red mb-3">1. 만든 이유</p>
			<p class="mb-3">2. 시연</p>
			<p class="mb-3">3. 코드 설명</p>
			<p>4. 느낀 점</p>
		</Slide>
		<Slide>
			<p class="font-bold mb-10">API 키 유출 상황</p>
			<center><img src="images/twitter-security.png" alt="https://www.boannews.com/media/view.asp?idx=108847"></center>
		</Slide>
		<Slide>
			<p class="font-bold mb-10">보안 취약점이 일어나는 이유</p>
			<center><img src="images/ibm-and-verizon.png" alt="https://www.varonis.com/blog/cybersecurity-statistics"></center>
			<p class="mt-5">74%의 보안 취약점은 사람의 문제로 발생됨</p>
			<p class="mt-5">보안 취약점을 알아차리는데 걸리는 시간은 207일</p>
		</Slide>
		<Slide>
			<p class="font-bold mb-10">보안 취약점 툴이 있음, 그러나</p>
			<center class="content-end">
				<img src="images/gitleaks.png" alt="https://github.com/gitleaks/gitleaks">
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
			<p class="mb-3 fragment highlight-red">2. 시연</p>
			<p class="mb-3">3. 코드 설명</p>
			<p>4. 느낀 점</p>
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
