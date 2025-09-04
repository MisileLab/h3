# Scalar 전투 시스템 구현

## 개요
Scalar 게임의 핵심 전투 시스템이 구현되었습니다. 이 시스템은 "진정한 승리는 적을 섬멸하는 것이 아니라, 모두 함께 집에 돌아가는 것이다"라는 철학을 바탕으로 설계되었습니다.

## 핵심 시스템

### 1. 기계 캐릭터 시스템
- **MechCharacter.cs**: 모든 기계의 기본 클래스
- **MechBodyPart.cs**: 부위별 손상 시스템
- **RexMech.cs**: 프론트라인 가디언 (탱커)
- **LunaMech.cs**: 테크니컬 서포터 (힐러/해커)
- **ZeroMech.cs**: 스피드 스카우트 (정찰/기동)
- **NovaMech.cs**: 헤비 어태커 (광역 딜러)

### 2. 턴제 전투 시스템
- **BattleSystem.cs**: 턴제 전투 관리
- **BattleActor.cs**: 전투 참가자 관리
- 속도 기반 턴 순서 결정
- AP(Action Point) 시스템
- 전투 종료 조건 관리

### 3. 협력 시스템
- **CooperationSystem.cs**: 동료 보호 중심의 협력 스킬
- 가드, 응급처치, 전술이동, 연계공격 등
- 신뢰도 시스템으로 협력 스킬 강화
- 기계 조합에 따른 특별 효과

### 4. 부위별 손상 시스템
- 머리/센서 (15%), 몸통/코어 (40%), 오른팔/주무장 (20%), 왼팔/보조장비 (15%), 다리/이동부 (10%)
- 부위 파괴 시 치명적인 패널티
- 단계별 손상 연출 (경미 → 중간 → 심각 → 기능정지)

### 5. 적 AI 시스템
- **EnemyAI.cs**: 다양한 적 타입 구현
- Scrapper, Sentinel, Interceptor, Enforcer
- 각각 고유한 행동 패턴과 전략
- 해킹, 억제, 도발 등 상태 효과

### 6. 대화 시스템
- **DialogueSystem.cs**: 전투 중 감성적 상호작용
- 상황별 대사 트리거
- 협력 행동에 따른 대사
- 신뢰도 증가에 따른 대사 변화

### 7. UI 시스템
- **BattleUI.cs**: 전투 인터페이스
- **MechStatusUI.cs**: 기계 상태 표시
- **BodyPartStatusUI.cs**: 부위별 상태 표시
- **CooperationOptionUI.cs**: 협력 옵션 선택

### 8. 게임 관리
- **GameManager.cs**: 전체 게임 상태 관리
- **BattleInitializer.cs**: 테스트용 전투 초기화

## 주요 특징

### 협력 중심의 전투
- 모든 전투 시스템이 '동료 보호'를 중심으로 설계
- 적 제거보다 아군 보호가 더 높은 가치
- 협력 행동에 더 큰 보상 제공

### 감성적 유대감 형성
- 기계들이 각자의 성격과 감정을 가진 '동료'
- 전투 중 상호작용 대사로 동료애 형성
- 신뢰도 시스템으로 관계 성장 체감

### 전략적 선택의 자유
- 다양한 기계 조합과 스킬 연계
- 회피, 협상, 후퇴 등 비폭력적 해결책
- 창의적인 전략 수립 가능

### 생존 우선주의
- 전투는 유일한 해결책이 아님
- 패배는 끝이 아닌 새로운 이야기의 시작
- 동료 구출과 재정비 시스템

## 사용 방법

### 1. 기본 설정
1. BattleInitializer를 씬에 배치
2. 기계 프리팹과 적 프리팹 설정
3. 스폰 포인트 설정
4. autoStartBattle을 true로 설정

### 2. 전투 시작
```csharp
GameManager.Instance.StartBattle();
```

### 3. 협력 스킬 사용
```csharp
CooperationSystem.Instance.UseCooperation(user, target, "가드");
```

### 4. 대화 표시
```csharp
DialogueSystem.Instance.ShowDialogue("렉스", "뒤로 물러나! 내가 막을게!");
```

## 확장 가능성

### 새로운 기계 타입 추가
1. MechCharacter를 상속받는 새 클래스 생성
2. MechType enum에 새 타입 추가
3. 고유 능력과 스탯 설정

### 새로운 협력 스킬 추가
1. CooperationType enum에 새 타입 추가
2. CooperationSystem에서 실행 로직 구현
3. UI에서 선택 가능하도록 설정

### 새로운 적 타입 추가
1. EnemyType enum에 새 타입 추가
2. EnemyAI에서 행동 패턴 구현
3. 고유 능력과 특성 설정

## 디버깅

### 로그 확인
- 모든 주요 이벤트는 Debug.Log로 출력
- 전투 진행 상황 추적 가능
- 협력 행동과 신뢰도 변화 확인

### 테스트 기능
- BattleInitializer의 Context Menu 사용
- "테스트 전투 시작"으로 즉시 테스트
- "기계들만 생성"으로 설정 확인

## 주의사항

1. 모든 기계는 MechCharacter를 상속받아야 함
2. UI 요소들은 반드시 연결되어야 함
3. 프리팹에는 적절한 컴포넌트가 부착되어야 함
4. 스폰 포인트는 충분한 수가 설정되어야 함

이 시스템은 Scalar 게임의 핵심 철학을 구현하며, 플레이어에게 감성적 몰입과 전략적 깊이를 동시에 제공합니다.
