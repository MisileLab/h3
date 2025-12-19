import { EpisodeStoryScript } from './types';

export const EP1_OUTSKIRTS_SCRIPT: EpisodeStoryScript = {
  episodeId: 'ep1',
  triggers: {
    TRG_BOOT_000: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'CASSETTE TACTICS / TERMINAL v0.9' },
        { speaker: 'SYSTEM', text: 'POWER: UNSTABLE' },
        { speaker: 'SYSTEM', text: 'MAGITECH BUS: NO SIGNAL' },
        { speaker: 'SYSTEM', text: 'BOOT SEQUENCE...' },
      ],
    },
    TRG_BOOT_010: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'AI CORE: I.O. (Initial Observer)' },
        { speaker: 'SYSTEM', text: 'STATUS: INITIALIZING' },
        { speaker: 'SYSTEM', text: 'SENSORS: OFFLINE' },
        { speaker: 'SYSTEM', text: 'MOTOR CONTROL: OFFLINE' },
        { speaker: 'SYSTEM', text: 'MEMORY: FRAGMENTED' },
      ],
    },
    TRG_BOOT_020: {
      once: 'session',
      lines: [
        { speaker: 'NIKO', text: '들려? …아, 좋아. 끊기지 말고.' },
        { speaker: 'NIKO', text: '난 니코. 공학동 생존자. 네가 깨어났다는 로그가 떴어.' },
        { speaker: 'NIKO', text: '움직일 몸이 필요하지? 드론 하나 붙였어. 네 눈은 이제 그거야.' },
      ],
    },
    TRG_BOOT_030: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'REMOTE VIEW: LINKED' },
        { speaker: 'SYSTEM', text: 'COCKPIT MODE: ACTIVE' },
      ],
    },
    TRG_BOOT_040: {
      once: 'session',
      lines: [
        { speaker: 'NIKO', text: '밖은… ‘공명 재해’ 이후로 엉망이야. 원인? 몰라.' },
        { speaker: 'NIKO', text: '그래도 규칙은 하나면 돼.' },
        { speaker: 'NIKO', text: 'Don\u2019t Panic. Organize.' },
        { speaker: 'SYSTEM', text: 'SLOGAN REGISTERED' },
      ],
    },

    TRG_N0_ENTER: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'NODE: GATEHOUSE' },
        { speaker: 'SYSTEM', text: 'THREAT: LOW' },
        { speaker: 'NIKO', text: '게이트하우스부터. 바깥 공기가 더 나빠지기 전에 길을 뚫자.' },
      ],
    },
    TRG_TUT_INTENT_FIRST: {
      once: 'session',
      lines: [
        { speaker: 'TUT', text: 'ENEMY INTENT ONLINE' },
        { speaker: 'TUT', text: '적의 다음 행동이 표시된다. 바뀌지 않는다.' },
        { speaker: 'NIKO', text: '보이지? 저 표시. 저건 예고야. 네가 움직이면, 피할 수 있어.' },
      ],
    },
    TRG_TUT_NORNG_FIRST: {
      once: 'session',
      lines: [
        { speaker: 'TUT', text: '명중률 없음. 확정 명중 / 확정 피해.' },
        { speaker: 'NIKO', text: '여긴 운이 없어. 계산만 남아.' },
      ],
    },
    TRG_TUT_TIMER_START: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'TURN TIMER: 30.0s' },
        { speaker: 'TUT', text: '플레이어 턴은 30초. 시간이 0이면 강제 종료된다.' },
        { speaker: 'NIKO', text: '생각은 깊게. 손은 빠르게.' },
      ],
    },
    TRG_TIMER_10S: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: '10s REMAINING' },
        { speaker: 'NIKO', text: '멈추지 마. 완벽 말고, 완료가 먼저야.' },
      ],
    },
    TRG_HALTED_FIRST: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'SYSTEM HALTED' },
        { speaker: 'TUT', text: '시간 초과로 턴이 종료됐다.' },
        { speaker: 'NIKO', text: '…괜찮아. 다음엔 더 빨라져.' },
      ],
    },
    TRG_LOOT_HINT: {
      once: 'session',
      lines: [
        { speaker: 'NIKO', text: '저 상자 봐? 열어.' },
        { speaker: 'NIKO', text: '드랍은 없어. 우리가 만들려면 재료가 필요해.' },
      ],
    },
    TRG_LOOT_OPEN_FIRST: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'CONTAINER OPENED' },
        { speaker: 'TUT', text: '전리품이 “트레이”로 올라왔다.' },
      ],
    },
    TRG_TRAY_EXPLAIN: {
      once: 'session',
      lines: [
        { speaker: 'TUT', text: '트레이는 전투 중에도 정리해야 한다.' },
        { speaker: 'NIKO', text: '전투는 머리로. 정리는 손으로. …둘 다 해.' },
      ],
    },
    TRG_TETRIS_EXPLAIN: {
      once: 'session',
      lines: [
        { speaker: 'TUT', text: '드래그로 인벤토리에 넣기. 회전: Q / E' },
        { speaker: 'NIKO', text: '공간은 제한돼. 맞춰 넣는 게 실력이야.' },
      ],
    },
    TRG_BUFFER_EXPLAIN: {
      once: 'session',
      lines: [
        { speaker: 'TUT', text: '급하면 “버퍼”로 보낼 수 있다. (우클릭/단축키)' },
        { speaker: 'NIKO', text: '버퍼는 임시야. 넘치면… 터진다.' },
      ],
    },
    TRG_BUFFER_OVERFLOW_FIRST: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'DATA SPILL' },
        { speaker: 'TUT', text: '버퍼 오버플로: 아이템 1개 손실 + HEAT 상승' },
        { speaker: 'NIKO', text: '봤지? 공짜는 없어. 잃는 건 너야.' },
      ],
    },
    TRG_HEAT_INTRO: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'HEAT: 0/10' },
        { speaker: 'TUT', text: 'HEAT가 높을수록 추출이 길어진다.' },
        { speaker: 'NIKO', text: '욕심내면 열이 오른다. 열이 오르면… 빠져나가기가 힘들어.' },
      ],
    },
    TRG_N0_CLEAR: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'THREAT CLEARED' },
        { speaker: 'NIKO', text: '좋아. 이제 변전소로 가자. 전력부터 살려야 해.' },
      ],
    },

    TRG_N1_ENTER: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'NODE: SUBSTATION' },
        { speaker: 'SYSTEM', text: 'OBJECTIVE: RESTORE POWER' },
        { speaker: 'NIKO', text: '여기가 변전소야. 전력만 돌아오면… 캠퍼스 장비를 살릴 수 있어.' },
      ],
    },
    TRG_STAB_GIVE: {
      once: 'session',
      lines: [
        { speaker: 'NIKO', text: '이거 받아.' },
        { speaker: 'SYSTEM', text: 'ITEM ACQUIRED: STABILIZER (Charge 1)' },
        { speaker: 'TUT', text: 'F : STABILIZER — 턴 타이머 +10초 / HEAT +1' },
        { speaker: 'NIKO', text: '급할 때 숨을 사는 버튼이야. 단, 공짜는 아니지.' },
      ],
    },
    TRG_STAB_USE_FIRST: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'STABILIZER ACTIVATED' },
        { speaker: 'SYSTEM', text: '+10.0s' },
        { speaker: 'SYSTEM', text: 'HEAT +1' },
        { speaker: 'NIKO', text: '좋아. 지금 그 10초로 정리해. 그리고 결정해.' },
      ],
    },
    TRG_JAMMER_SPAWN_FIRST: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'THREAT DETECTED: JAMMER' },
        { speaker: 'TUT', text: 'JAMMER 생존 중: 버퍼 용량 -1' },
        { speaker: 'NIKO', text: '저놈이 신호를 망쳐. …네 손도 꼬이게 만들지.' },
      ],
    },
    TRG_JAMMER_EMP_HIT_FIRST: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'EMP HIT' },
        { speaker: 'SYSTEM', text: 'HEAT +1' },
        { speaker: 'NIKO', text: '맞았어. 열이 올라간다. …당황하지 마.' },
      ],
    },
    TRG_BREAKER_SEEN: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'BREAKER PANEL ONLINE' },
        { speaker: 'TUT', text: '브레이커 업로드: 유지(턴 종료) ×2' },
        { speaker: 'NIKO', text: '패널에 패치를 업로드해. 두 번이면 돼.' },
      ],
    },
    TRG_BREAKER_UPLOAD_1: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'UPLOAD: 1/2' },
        { speaker: 'NIKO', text: '좋아. 한 번 더.' },
      ],
    },
    TRG_BREAKER_UPLOAD_2: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'UPLOAD: 2/2' },
        { speaker: 'SYSTEM', text: 'POWER RESTORED' },
        { speaker: 'NIKO', text: '왔다… 전력 살아났어.' },
        { speaker: 'NIKO', text: '정비 야드. 이제 만드는 시간이야.' },
        { speaker: 'SYSTEM', text: 'BLUEPRINT VIEW ONLINE' },
      ],
    },

    TRG_N2_ENTER: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'NODE: MAINTENANCE YARD' },
        { speaker: 'SYSTEM', text: 'SAFE ZONE' },
        { speaker: 'NIKO', text: '여긴 잠깐 안전해. 숨 돌려. 그리고… 생산하자.' },
      ],
    },
    TRG_BLUEPRINT_INTRO: {
      once: 'session',
      lines: [
        { speaker: 'TUT', text: 'Blueprint View: 건물을 배치하고 라인을 연결해 생산한다.' },
        { speaker: 'NIKO', text: '우린 줍는 게 아니라 만드는 쪽이야.' },
      ],
    },
    TRG_PLACE_EXTRACTOR: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'PLACED: EXTRACTOR' },
        { speaker: 'NIKO', text: '채집기부터. 재료가 들어와야 아무것도 못 해.' },
      ],
    },
    TRG_PLACE_REFINER: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'PLACED: REFINER' },
        { speaker: 'NIKO', text: '정제기. 잡동사니를 쓸모 있게 바꿔.' },
      ],
    },
    TRG_PLACE_PRESS: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'PLACED: PRESS' },
        { speaker: 'NIKO', text: '프레스… 이제 네가 쓸 카세트를 찍어내는 거야.' },
      ],
    },
    TRG_LINE_CONNECTED: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'LINE STATUS: CONNECTED' },
        { speaker: 'NIKO', text: '좋아. 흐름이 생겼어. 이제 돌려.' },
      ],
    },
    TRG_PRODUCE_CASSETTE_FIRST: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'PRODUCED: BASIC ATTACK CASSETTE' },
        { speaker: 'NIKO', text: '네 첫 카세트다. 네가 만든 첫 “주문”이지.' },
      ],
    },
    TRG_PRODUCE_KIT_FIRST: {
      once: 'session',
      lines: [
        { speaker: 'SYSTEM', text: 'PRODUCED: PATCH KIT' },
        { speaker: 'NIKO', text: '그리고 이건 수리 키트. 오래 살려면 필수야.' },
      ],
    },
    TRG_N2_OBJECTIVE_COMPLETE: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'PRODUCTION OBJECTIVE COMPLETE' },
        { speaker: 'NIKO', text: '됐어. 이제 나가자. …선택이 있다.' },
      ],
    },
    TRG_EXTRACT_CHOICE: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'EXTRACTION OPTIONS AVAILABLE' },
        { speaker: 'NIKO', text: '가까운 길은 안전해. 하지만 얻는 건 적지.' },
        { speaker: 'NIKO', text: '먼 길은 위험해. 대신… 더 많은 걸 건질 수 있어.' },
        { speaker: 'TUT', text: 'Extract A(안전/보상↓) / Extract B(위험/보상↑)' },
      ],
    },

    TRG_N3A_ENTER: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'NODE: GATEHOUSE UPLINK' },
        { speaker: 'SYSTEM', text: 'UPLOAD REQUIRED: 2 (+1 if HEAT>=4)' },
        { speaker: 'NIKO', text: '여기서 끝낼 수 있어. 일단 살아나가자.' },
      ],
    },
    TRG_UPLOAD_START_A: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'UPLOAD STARTED' },
        { speaker: 'TUT', text: '추출 존(3×3)을 유지하면 턴 종료마다 업로드가 진행된다.' },
        { speaker: 'NIKO', text: '존을 유지해. 나가면 끊겨.' },
      ],
    },
    TRG_UPLOAD_STEP_A_1: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'UPLOAD: 1/2' },
        { speaker: 'NIKO', text: '계속 유지.' },
      ],
    },
    TRG_UPLOAD_HEAT_WARN: {
      once: 'run',
      lines: [{ speaker: 'NIKO', text: 'HEAT가 높아. 업로드가 더 오래 걸릴 거야.' }],
    },
    TRG_UPLOAD_COMPLETE_A: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'UPLOAD COMPLETE' },
        { speaker: 'NIKO', text: '좋아. 나왔다.' },
      ],
    },

    TRG_N3B_ENTER: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'NODE: RELAY TOWER UPLINK' },
        { speaker: 'SYSTEM', text: 'UPLOAD REQUIRED: 3 (+1 if HEAT>=4)' },
        { speaker: 'NIKO', text: '여긴 멀어. 그리고 시끄러워. …하지만 값어치가 있어.' },
      ],
    },
    TRG_UPLOAD_START_B: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'UPLOAD STARTED' },
        { speaker: 'TUT', text: '업로드는 턴 종료마다 진행된다. 존을 유지하라.' },
        { speaker: 'NIKO', text: '침착하게. 정리하고. 버티고.' },
      ],
    },
    TRG_UPLOAD_STEP_B_1: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'UPLOAD: 1/3' },
        { speaker: 'NIKO', text: '좋아. 아직이야.' },
      ],
    },
    TRG_UPLOAD_STEP_B_2: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'UPLOAD: 2/3' },
        { speaker: 'NIKO', text: '…잠깐.' },
        { speaker: 'NIKO', text: '잡음 속에… 신호가 있어.' },
      ],
    },
    TRG_RESCUE_THREAD_UNLOCK: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'RESCUE THREAD 01 UNLOCKED' },
        { speaker: 'SYSTEM', text: 'UNKNOWN STUDENT SIGNAL CAPTURED' },
        { speaker: 'NIKO', text: '사람 목소리… 맞아. 아직 살아있어.' },
        { speaker: 'NIKO', text: '다음엔 저 신호를 따라가자.' },
      ],
    },
    TRG_UPLOAD_HEAT_WARN_B: {
      once: 'run',
      lines: [{ speaker: 'NIKO', text: '열이 높아. 오래 걸린다. 버텨.' }],
    },
    TRG_UPLOAD_COMPLETE_B: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'UPLOAD COMPLETE' },
        { speaker: 'NIKO', text: '좋아. 나왔다. …그리고 잡았다.' },
      ],
    },

    TRG_RUN_COMPLETE: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'RUN COMPLETE' },
        { speaker: 'SYSTEM', text: 'DATA SYNCHRONIZED' },
        { speaker: 'SYSTEM', text: 'BLUEPRINTS UPDATED' },
        { speaker: 'NIKO', text: '네가 모은 건 남아. 다음엔… 더 침착해질 수 있어.' },
      ],
    },
    TRG_REWARD_SUMMARY_A: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'REWARD: COMBAT DATA +2' },
        { speaker: 'SYSTEM', text: 'REWARD: FACTORY DATA +2' },
        { speaker: 'SYSTEM', text: 'REWARD: LOG DATA +1' },
        { speaker: 'NIKO', text: '오늘은 여기까지. 내일은… 더 멀리.' },
      ],
    },
    TRG_REWARD_SUMMARY_B: {
      once: 'run',
      lines: [
        { speaker: 'SYSTEM', text: 'REWARD: COMBAT DATA +3' },
        { speaker: 'SYSTEM', text: 'REWARD: FACTORY DATA +2' },
        { speaker: 'SYSTEM', text: 'REWARD: LOG DATA +3' },
        { speaker: 'SYSTEM', text: 'BONUS: RESCUE THREAD 01' },
        { speaker: 'NIKO', text: '신호가 잡혔어. …우린 혼자가 아니야.' },
      ],
    },
  },
};
