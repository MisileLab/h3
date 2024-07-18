export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set(["acting.png","ecoProgressAfter.png","ecoProgressBefore.png","ecoProgressFull.png","favicon.png","item.png","itemUseAfter.png","itemUseBefore.png","kickButton.png","thinking.png"]),
	mimeTypes: {".png":"image/png"},
	_: {
		client: {"start":"_app/immutable/entry/start.HarLD-op.js","app":"_app/immutable/entry/app.CwPfcjms.js","imports":["_app/immutable/entry/start.HarLD-op.js","_app/immutable/chunks/entry.B0Jmh7gx.js","_app/immutable/chunks/runtime.4Ab5sdzz.js","_app/immutable/entry/app.CwPfcjms.js","_app/immutable/chunks/preload-helper.DnUxZNQJ.js","_app/immutable/chunks/runtime.4Ab5sdzz.js","_app/immutable/chunks/disclose-version.Dle8pKw4.js","_app/immutable/chunks/render.rStpzCLw.js","_app/immutable/chunks/events.CjL9_RZI.js","_app/immutable/chunks/svelte-head.CJOXNnCy.js"],"stylesheets":[],"fonts":[],"uses_env_dynamic_public":false},
		nodes: [
			__memo(() => import('../output/server/nodes/0.js')),
			__memo(() => import('../output/server/nodes/1.js')),
			__memo(() => import('../output/server/nodes/2.js'))
		],
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
