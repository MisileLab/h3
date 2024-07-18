import { d as define_property, i as is_array, H as HYDRATION_START, a as HYDRATION_ERROR, b as HYDRATION_END, c as array_from, P as PassiveDelegatedEvents, r as render, p as push$1, s as setContext, e as pop$1 } from "./index.js";
let base = "";
let assets = base;
const initial = { base, assets };
function override(paths) {
  base = paths.base;
  assets = paths.assets;
}
function reset() {
  base = initial.base;
  assets = initial.assets;
}
function set_assets(path) {
  assets = initial.assets = path;
}
let public_env = {};
let safe_public_env = {};
function set_private_env(environment) {
}
function set_public_env(environment) {
  public_env = environment;
}
function set_safe_public_env(environment) {
  safe_public_env = environment;
}
function equals(value) {
  return value === this.v;
}
function safe_not_equal(a, b) {
  return a != a ? b == b : a !== b || a !== null && typeof a === "object" || typeof a === "function";
}
function safe_equals(value) {
  return !safe_not_equal(value, this.v);
}
const DERIVED = 1 << 1;
const EFFECT = 1 << 2;
const RENDER_EFFECT = 1 << 3;
const BLOCK_EFFECT = 1 << 4;
const BRANCH_EFFECT = 1 << 5;
const ROOT_EFFECT = 1 << 6;
const UNOWNED = 1 << 7;
const DISCONNECTED = 1 << 8;
const CLEAN = 1 << 9;
const DIRTY = 1 << 10;
const MAYBE_DIRTY = 1 << 11;
const INERT = 1 << 12;
const DESTROYED = 1 << 13;
const EFFECT_RAN = 1 << 14;
const HEAD_EFFECT = 1 << 18;
function effect_update_depth_exceeded() {
  {
    throw new Error("effect_update_depth_exceeded");
  }
}
function hydration_failed() {
  {
    throw new Error("hydration_failed");
  }
}
function state_unsafe_mutation() {
  {
    throw new Error("state_unsafe_mutation");
  }
}
function push_effect(effect2, parent_effect) {
  var parent_last = parent_effect.last;
  if (parent_last === null) {
    parent_effect.last = parent_effect.first = effect2;
  } else {
    parent_last.next = effect2;
    effect2.prev = parent_last;
    parent_effect.last = effect2;
  }
}
function create_effect(type, fn, sync, push2 = true) {
  var is_root = (type & ROOT_EFFECT) !== 0;
  var effect2 = {
    ctx: current_component_context,
    deps: null,
    nodes: null,
    f: type | DIRTY,
    first: null,
    fn,
    last: null,
    next: null,
    parent: is_root ? null : current_effect,
    prev: null,
    teardown: null,
    transitions: null,
    version: 0
  };
  if (sync) {
    var previously_flushing_effect = is_flushing_effect;
    try {
      set_is_flushing_effect(true);
      update_effect(effect2);
      effect2.f |= EFFECT_RAN;
    } catch (e) {
      destroy_effect(effect2);
      throw e;
    } finally {
      set_is_flushing_effect(previously_flushing_effect);
    }
  } else if (fn !== null) {
    schedule_effect(effect2);
  }
  var inert = sync && effect2.deps === null && effect2.first === null && effect2.nodes === null && effect2.teardown === null;
  if (!inert && !is_root && push2) {
    if (current_effect !== null) {
      push_effect(effect2, current_effect);
    }
    if (current_reaction !== null && (current_reaction.f & DERIVED) !== 0) {
      push_effect(effect2, current_reaction);
    }
  }
  return effect2;
}
function effect_root(fn) {
  const effect2 = create_effect(ROOT_EFFECT, fn, true);
  return () => {
    destroy_effect(effect2);
  };
}
function effect(fn) {
  return create_effect(EFFECT, fn, false);
}
function branch(fn, push2 = true) {
  return create_effect(RENDER_EFFECT | BRANCH_EFFECT, fn, true, push2);
}
function execute_effect_teardown(effect2) {
  var teardown = effect2.teardown;
  if (teardown !== null) {
    const previous_reaction = current_reaction;
    set_current_reaction(null);
    try {
      teardown.call(null);
    } finally {
      set_current_reaction(previous_reaction);
    }
  }
}
function destroy_effect(effect2, remove_dom = true) {
  var removed = false;
  if ((remove_dom || (effect2.f & HEAD_EFFECT) !== 0) && effect2.nodes !== null) {
    var node = effect2.nodes.start;
    var end = effect2.nodes.end;
    while (node !== null) {
      var next = node === end ? null : (
        /** @type {import('#client').TemplateNode} */
        node.nextSibling
      );
      node.remove();
      node = next;
    }
    removed = true;
  }
  destroy_effect_children(effect2, remove_dom && !removed);
  remove_reactions(effect2, 0);
  set_signal_status(effect2, DESTROYED);
  if (effect2.transitions) {
    for (const transition of effect2.transitions) {
      transition.stop();
    }
  }
  execute_effect_teardown(effect2);
  var parent = effect2.parent;
  if (parent !== null && (effect2.f & BRANCH_EFFECT) !== 0 && parent.first !== null) {
    unlink_effect(effect2);
  }
  effect2.next = effect2.prev = effect2.teardown = effect2.ctx = effect2.deps = effect2.parent = effect2.fn = effect2.nodes = null;
}
function unlink_effect(effect2) {
  var parent = effect2.parent;
  var prev = effect2.prev;
  var next = effect2.next;
  if (prev !== null) prev.next = next;
  if (next !== null) next.prev = prev;
  if (parent !== null) {
    if (parent.first === effect2) parent.first = next;
    if (parent.last === effect2) parent.last = prev;
  }
}
function flush_tasks() {
}
function hydration_mismatch(location) {
  {
    console.warn("hydration_mismatch");
  }
}
function destroy_derived_children(derived) {
  destroy_effect_children(derived);
  var deriveds = derived.deriveds;
  if (deriveds !== null) {
    derived.deriveds = null;
    for (var i = 0; i < deriveds.length; i += 1) {
      destroy_derived(deriveds[i]);
    }
  }
}
function update_derived(derived) {
  destroy_derived_children(derived);
  var value = update_reaction(derived);
  var status = (current_skip_reaction || (derived.f & UNOWNED) !== 0) && derived.deps !== null ? MAYBE_DIRTY : CLEAN;
  set_signal_status(derived, status);
  if (!derived.equals(value)) {
    derived.v = value;
    derived.version = increment_version();
  }
}
function destroy_derived(signal) {
  destroy_derived_children(signal);
  remove_reactions(signal, 0);
  set_signal_status(signal, DESTROYED);
  signal.first = signal.last = signal.deps = signal.reactions = // @ts-expect-error `signal.fn` cannot be `null` while the signal is alive
  signal.fn = null;
}
const FLUSH_MICROTASK = 0;
const FLUSH_SYNC = 1;
let current_scheduler_mode = FLUSH_MICROTASK;
let is_micro_task_queued = false;
let is_flushing_effect = false;
function set_is_flushing_effect(value) {
  is_flushing_effect = value;
}
let current_queued_root_effects = [];
let flush_count = 0;
let current_reaction = null;
function set_current_reaction(reaction) {
  current_reaction = reaction;
}
let current_effect = null;
let new_deps = null;
let skipped_deps = 0;
let current_untracked_writes = null;
function set_current_untracked_writes(value) {
  current_untracked_writes = value;
}
let current_version = 0;
let current_skip_reaction = false;
let current_component_context = null;
function increment_version() {
  return current_version++;
}
function is_runes() {
  return current_component_context !== null && current_component_context.l === null;
}
function check_dirtiness(reaction) {
  var flags = reaction.f;
  if ((flags & DIRTY) !== 0) {
    return true;
  }
  if ((flags & MAYBE_DIRTY) !== 0) {
    var dependencies = reaction.deps;
    if (dependencies !== null) {
      var is_unowned = (flags & UNOWNED) !== 0;
      var i;
      if ((flags & DISCONNECTED) !== 0) {
        for (i = 0; i < dependencies.length; i++) {
          (dependencies[i].reactions ??= []).push(reaction);
        }
        reaction.f ^= DISCONNECTED;
      }
      for (i = 0; i < dependencies.length; i++) {
        var dependency = dependencies[i];
        if (check_dirtiness(
          /** @type {import('#client').Derived} */
          dependency
        )) {
          update_derived(
            /** @type {import('#client').Derived} */
            dependency
          );
        }
        if (dependency.version > reaction.version) {
          return true;
        }
        if (is_unowned) {
          if (!current_skip_reaction && !dependency?.reactions?.includes(reaction)) {
            (dependency.reactions ??= []).push(reaction);
          }
        }
      }
    }
    set_signal_status(reaction, CLEAN);
  }
  return false;
}
function handle_error(error, effect2, component_context) {
  {
    throw error;
  }
}
function update_reaction(reaction) {
  var previous_deps = new_deps;
  var previous_skipped_deps = skipped_deps;
  var previous_untracked_writes = current_untracked_writes;
  var previous_reaction = current_reaction;
  var previous_skip_reaction = current_skip_reaction;
  new_deps = /** @type {null | import('#client').Value[]} */
  null;
  skipped_deps = 0;
  current_untracked_writes = null;
  current_reaction = (reaction.f & (BRANCH_EFFECT | ROOT_EFFECT)) === 0 ? reaction : null;
  current_skip_reaction = !is_flushing_effect && (reaction.f & UNOWNED) !== 0;
  try {
    var result = (
      /** @type {Function} */
      (0, reaction.fn)()
    );
    var deps = reaction.deps;
    if (new_deps !== null) {
      var dependency;
      var i;
      if (deps !== null) {
        var array = skipped_deps === 0 ? new_deps : deps.slice(0, skipped_deps).concat(new_deps);
        var set2 = array.length > 16 ? new Set(array) : null;
        for (i = skipped_deps; i < deps.length; i++) {
          dependency = deps[i];
          if (set2 !== null ? !set2.has(dependency) : !array.includes(dependency)) {
            remove_reaction(reaction, dependency);
          }
        }
      }
      if (deps !== null && skipped_deps > 0) {
        deps.length = skipped_deps + new_deps.length;
        for (i = 0; i < new_deps.length; i++) {
          deps[skipped_deps + i] = new_deps[i];
        }
      } else {
        reaction.deps = deps = new_deps;
      }
      if (!current_skip_reaction) {
        for (i = skipped_deps; i < deps.length; i++) {
          dependency = deps[i];
          var reactions = dependency.reactions;
          if (reactions === null) {
            dependency.reactions = [reaction];
          } else if (reactions[reactions.length - 1] !== reaction && !reactions.includes(reaction)) {
            reactions.push(reaction);
          }
        }
      }
    } else if (deps !== null && skipped_deps < deps.length) {
      remove_reactions(reaction, skipped_deps);
      deps.length = skipped_deps;
    }
    return result;
  } finally {
    new_deps = previous_deps;
    skipped_deps = previous_skipped_deps;
    current_untracked_writes = previous_untracked_writes;
    current_reaction = previous_reaction;
    current_skip_reaction = previous_skip_reaction;
  }
}
function remove_reaction(signal, dependency) {
  const reactions = dependency.reactions;
  let reactions_length = 0;
  if (reactions !== null) {
    reactions_length = reactions.length - 1;
    const index = reactions.indexOf(signal);
    if (index !== -1) {
      if (reactions_length === 0) {
        dependency.reactions = null;
      } else {
        reactions[index] = reactions[reactions_length];
        reactions.pop();
      }
    }
  }
  if (reactions_length === 0 && (dependency.f & DERIVED) !== 0) {
    set_signal_status(dependency, MAYBE_DIRTY);
    if ((dependency.f & (UNOWNED | DISCONNECTED)) === 0) {
      dependency.f ^= DISCONNECTED;
    }
    remove_reactions(
      /** @type {import('#client').Derived} **/
      dependency,
      0
    );
  }
}
function remove_reactions(signal, start_index) {
  var dependencies = signal.deps;
  if (dependencies === null) return;
  var active_dependencies = start_index === 0 ? null : dependencies.slice(0, start_index);
  var seen = /* @__PURE__ */ new Set();
  for (var i = start_index; i < dependencies.length; i++) {
    var dependency = dependencies[i];
    if (seen.has(dependency)) continue;
    seen.add(dependency);
    if (active_dependencies === null || !active_dependencies.includes(dependency)) {
      remove_reaction(signal, dependency);
    }
  }
}
function destroy_effect_children(signal, remove_dom = false) {
  var effect2 = signal.first;
  signal.first = signal.last = null;
  while (effect2 !== null) {
    var next = effect2.next;
    destroy_effect(effect2, remove_dom);
    effect2 = next;
  }
}
function update_effect(effect2) {
  var flags = effect2.f;
  if ((flags & DESTROYED) !== 0) {
    return;
  }
  set_signal_status(effect2, CLEAN);
  var component_context = effect2.ctx;
  var previous_effect = current_effect;
  var previous_component_context = current_component_context;
  current_effect = effect2;
  current_component_context = component_context;
  try {
    if ((flags & BLOCK_EFFECT) === 0) {
      destroy_effect_children(effect2);
    }
    execute_effect_teardown(effect2);
    var teardown = update_reaction(effect2);
    effect2.teardown = typeof teardown === "function" ? teardown : null;
    effect2.version = current_version;
  } catch (error) {
    handle_error(
      /** @type {Error} */
      error
    );
  } finally {
    current_effect = previous_effect;
    current_component_context = previous_component_context;
  }
}
function infinite_loop_guard() {
  if (flush_count > 1e3) {
    flush_count = 0;
    effect_update_depth_exceeded();
  }
  flush_count++;
}
function flush_queued_root_effects(root_effects) {
  var length = root_effects.length;
  if (length === 0) {
    return;
  }
  infinite_loop_guard();
  var previously_flushing_effect = is_flushing_effect;
  is_flushing_effect = true;
  try {
    for (var i = 0; i < length; i++) {
      var effect2 = root_effects[i];
      if (effect2.first === null && (effect2.f & BRANCH_EFFECT) === 0) {
        flush_queued_effects([effect2]);
      } else {
        var collected_effects = [];
        process_effects(effect2, collected_effects);
        flush_queued_effects(collected_effects);
      }
    }
  } finally {
    is_flushing_effect = previously_flushing_effect;
  }
}
function flush_queued_effects(effects) {
  var length = effects.length;
  if (length === 0) return;
  for (var i = 0; i < length; i++) {
    var effect2 = effects[i];
    if ((effect2.f & (DESTROYED | INERT)) === 0 && check_dirtiness(effect2)) {
      update_effect(effect2);
      if (effect2.deps === null && effect2.first === null && effect2.nodes === null) {
        if (effect2.teardown === null) {
          unlink_effect(effect2);
        } else {
          effect2.fn = null;
        }
      }
    }
  }
}
function process_deferred() {
  is_micro_task_queued = false;
  if (flush_count > 1001) {
    return;
  }
  const previous_queued_root_effects = current_queued_root_effects;
  current_queued_root_effects = [];
  flush_queued_root_effects(previous_queued_root_effects);
  if (!is_micro_task_queued) {
    flush_count = 0;
  }
}
function schedule_effect(signal) {
  if (current_scheduler_mode === FLUSH_MICROTASK) {
    if (!is_micro_task_queued) {
      is_micro_task_queued = true;
      queueMicrotask(process_deferred);
    }
  }
  var effect2 = signal;
  while (effect2.parent !== null) {
    effect2 = effect2.parent;
    var flags = effect2.f;
    if ((flags & BRANCH_EFFECT) !== 0) {
      if ((flags & CLEAN) === 0) return;
      set_signal_status(effect2, MAYBE_DIRTY);
    }
  }
  current_queued_root_effects.push(effect2);
}
function process_effects(effect2, collected_effects) {
  var current_effect2 = effect2.first;
  var effects = [];
  main_loop: while (current_effect2 !== null) {
    var flags = current_effect2.f;
    var is_active = (flags & (DESTROYED | INERT)) === 0;
    var is_branch = flags & BRANCH_EFFECT;
    var is_clean = (flags & CLEAN) !== 0;
    var child = current_effect2.first;
    if (is_active && (!is_branch || !is_clean)) {
      if (is_branch) {
        set_signal_status(current_effect2, CLEAN);
      }
      if ((flags & RENDER_EFFECT) !== 0) {
        if (!is_branch && check_dirtiness(current_effect2)) {
          update_effect(current_effect2);
          child = current_effect2.first;
        }
        if (child !== null) {
          current_effect2 = child;
          continue;
        }
      } else if ((flags & EFFECT) !== 0) {
        if (is_branch || is_clean) {
          if (child !== null) {
            current_effect2 = child;
            continue;
          }
        } else {
          effects.push(current_effect2);
        }
      }
    }
    var sibling = current_effect2.next;
    if (sibling === null) {
      let parent = current_effect2.parent;
      while (parent !== null) {
        if (effect2 === parent) {
          break main_loop;
        }
        var parent_sibling = parent.next;
        if (parent_sibling !== null) {
          current_effect2 = parent_sibling;
          continue main_loop;
        }
        parent = parent.parent;
      }
    }
    current_effect2 = sibling;
  }
  for (var i = 0; i < effects.length; i++) {
    child = effects[i];
    collected_effects.push(child);
    process_effects(child, collected_effects);
  }
}
function flush_sync(fn, flush_previous = true) {
  var previous_scheduler_mode = current_scheduler_mode;
  var previous_queued_root_effects = current_queued_root_effects;
  try {
    infinite_loop_guard();
    const root_effects = [];
    current_scheduler_mode = FLUSH_SYNC;
    current_queued_root_effects = root_effects;
    is_micro_task_queued = false;
    if (flush_previous) {
      flush_queued_root_effects(previous_queued_root_effects);
    }
    var result = fn?.();
    flush_tasks();
    if (current_queued_root_effects.length > 0 || root_effects.length > 0) {
      flush_sync();
    }
    flush_count = 0;
    return result;
  } finally {
    current_scheduler_mode = previous_scheduler_mode;
    current_queued_root_effects = previous_queued_root_effects;
  }
}
function get(signal) {
  var flags = signal.f;
  if ((flags & DESTROYED) !== 0) {
    return signal.v;
  }
  if (current_reaction !== null) {
    var deps = current_reaction.deps;
    if (new_deps === null && deps !== null && deps[skipped_deps] === signal) {
      skipped_deps++;
    } else if (deps === null || skipped_deps === 0 || deps[skipped_deps - 1] !== signal) {
      if (new_deps === null) {
        new_deps = [signal];
      } else if (new_deps[new_deps.length - 1] !== signal) {
        new_deps.push(signal);
      }
    }
    if (current_untracked_writes !== null && current_effect !== null && (current_effect.f & CLEAN) !== 0 && (current_effect.f & BRANCH_EFFECT) === 0 && current_untracked_writes.includes(signal)) {
      set_signal_status(current_effect, DIRTY);
      schedule_effect(current_effect);
    }
  }
  if ((flags & DERIVED) !== 0) {
    var derived = (
      /** @type {import('#client').Derived} */
      signal
    );
    if (check_dirtiness(derived)) {
      update_derived(derived);
    }
  }
  return signal.v;
}
const STATUS_MASK = ~(DIRTY | MAYBE_DIRTY | CLEAN);
function set_signal_status(signal, status) {
  signal.f = signal.f & STATUS_MASK | status;
}
function push(props, runes = false, fn) {
  current_component_context = {
    p: current_component_context,
    c: null,
    e: null,
    m: false,
    s: props,
    x: null,
    l: null
  };
  if (!runes) {
    current_component_context.l = {
      s: null,
      u: null,
      r1: [],
      r2: /* @__PURE__ */ source(false)
    };
  }
}
function pop(component) {
  const context_stack_item = current_component_context;
  if (context_stack_item !== null) {
    const effects = context_stack_item.e;
    if (effects !== null) {
      context_stack_item.e = null;
      for (var i = 0; i < effects.length; i++) {
        effect(effects[i]);
      }
    }
    current_component_context = context_stack_item.p;
    context_stack_item.m = true;
  }
  return (
    /** @type {T} */
    {}
  );
}
// @__NO_SIDE_EFFECTS__
function source(v) {
  return {
    f: 0,
    // TODO ideally we could skip this altogether, but it causes type errors
    v,
    reactions: null,
    equals,
    version: 0
  };
}
// @__NO_SIDE_EFFECTS__
function mutable_source(initial_value) {
  const s = /* @__PURE__ */ source(initial_value);
  s.equals = safe_equals;
  if (current_component_context !== null && current_component_context.l !== null) {
    (current_component_context.l.s ??= []).push(s);
  }
  return s;
}
function set(source2, value) {
  if (current_reaction !== null && is_runes() && (current_reaction.f & DERIVED) !== 0) {
    state_unsafe_mutation();
  }
  if (!source2.equals(value)) {
    source2.v = value;
    source2.version = increment_version();
    mark_reactions(source2, DIRTY);
    if (is_runes() && current_effect !== null && (current_effect.f & CLEAN) !== 0 && (current_effect.f & BRANCH_EFFECT) === 0) {
      if (new_deps !== null && new_deps.includes(source2)) {
        set_signal_status(current_effect, DIRTY);
        schedule_effect(current_effect);
      } else {
        if (current_untracked_writes === null) {
          set_current_untracked_writes([source2]);
        } else {
          current_untracked_writes.push(source2);
        }
      }
    }
  }
  return value;
}
function mark_reactions(signal, status) {
  var reactions = signal.reactions;
  if (reactions === null) return;
  var runes = is_runes();
  var length = reactions.length;
  for (var i = 0; i < length; i++) {
    var reaction = reactions[i];
    var flags = reaction.f;
    if ((flags & DIRTY) !== 0) continue;
    if (!runes && reaction === current_effect) continue;
    set_signal_status(reaction, status);
    if ((flags & (CLEAN | UNOWNED)) !== 0) {
      if ((flags & DERIVED) !== 0) {
        mark_reactions(
          /** @type {import('#client').Derived} */
          reaction,
          MAYBE_DIRTY
        );
      } else {
        schedule_effect(
          /** @type {import('#client').Effect} */
          reaction
        );
      }
    }
  }
}
let hydrating = false;
function set_hydrating(value) {
  hydrating = value;
}
let hydrate_node;
function set_hydrate_node(node) {
  return hydrate_node = node;
}
function hydrate_next() {
  return hydrate_node = /** @type {TemplateNode} */
  hydrate_node.nextSibling;
}
var $window;
function init_operations() {
  if ($window !== void 0) {
    return;
  }
  $window = window;
  var element_prototype = Element.prototype;
  element_prototype.__click = void 0;
  element_prototype.__className = "";
  element_prototype.__attributes = null;
  element_prototype.__e = void 0;
  Text.prototype.__t = void 0;
}
function empty() {
  return document.createTextNode("");
}
function clear_text_content(node) {
  node.textContent = "";
}
const all_registered_events = /* @__PURE__ */ new Set();
const root_event_handles = /* @__PURE__ */ new Set();
function handle_event_propagation(event) {
  var handler_element = this;
  var owner_document = (
    /** @type {Node} */
    handler_element.ownerDocument
  );
  var event_name = event.type;
  var path = event.composedPath?.() || [];
  var current_target = (
    /** @type {null | Element} */
    path[0] || event.target
  );
  var path_idx = 0;
  var handled_at = event.__root;
  if (handled_at) {
    var at_idx = path.indexOf(handled_at);
    if (at_idx !== -1 && (handler_element === document || handler_element === /** @type {any} */
    window)) {
      event.__root = handler_element;
      return;
    }
    var handler_idx = path.indexOf(handler_element);
    if (handler_idx === -1) {
      return;
    }
    if (at_idx <= handler_idx) {
      path_idx = at_idx;
    }
  }
  current_target = /** @type {Element} */
  path[path_idx] || event.target;
  if (current_target === handler_element) return;
  define_property(event, "currentTarget", {
    configurable: true,
    get() {
      return current_target || owner_document;
    }
  });
  try {
    var throw_error;
    var other_errors = [];
    while (current_target !== null) {
      var parent_element = current_target.parentNode || /** @type {any} */
      current_target.host || null;
      try {
        var delegated = current_target["__" + event_name];
        if (delegated !== void 0 && !/** @type {any} */
        current_target.disabled) {
          if (is_array(delegated)) {
            var [fn, ...data] = delegated;
            fn.apply(current_target, [event, ...data]);
          } else {
            delegated.call(current_target, event);
          }
        }
      } catch (error) {
        if (throw_error) {
          other_errors.push(error);
        } else {
          throw_error = error;
        }
      }
      if (event.cancelBubble || parent_element === handler_element || parent_element === null) {
        break;
      }
      current_target = parent_element;
    }
    if (throw_error) {
      for (let error of other_errors) {
        queueMicrotask(() => {
          throw error;
        });
      }
      throw throw_error;
    }
  } finally {
    event.__root = handler_element;
    current_target = handler_element;
  }
}
function assign_nodes(start, end) {
  current_effect.nodes ??= { start, end };
}
function mount(component, options2) {
  const anchor = options2.anchor ?? options2.target.appendChild(empty());
  return flush_sync(() => _mount(component, { ...options2, anchor }), false);
}
function hydrate(component, options2) {
  options2.intro = options2.intro ?? false;
  const target = options2.target;
  const was_hydrating = hydrating;
  try {
    return flush_sync(() => {
      var anchor = (
        /** @type {import('#client').TemplateNode} */
        target.firstChild
      );
      while (anchor && (anchor.nodeType !== 8 || /** @type {Comment} */
      anchor.data !== HYDRATION_START)) {
        anchor = /** @type {import('#client').TemplateNode} */
        anchor.nextSibling;
      }
      if (!anchor) {
        throw HYDRATION_ERROR;
      }
      set_hydrating(true);
      set_hydrate_node(
        /** @type {Comment} */
        anchor
      );
      hydrate_next();
      const instance = _mount(component, { ...options2, anchor });
      if (hydrate_node.nodeType !== 8 || /** @type {Comment} */
      hydrate_node.data !== HYDRATION_END) {
        hydration_mismatch();
        throw HYDRATION_ERROR;
      }
      set_hydrating(false);
      return instance;
    }, false);
  } catch (error) {
    if (error === HYDRATION_ERROR) {
      if (options2.recover === false) {
        hydration_failed();
      }
      init_operations();
      clear_text_content(target);
      set_hydrating(false);
      return mount(component, options2);
    }
    throw error;
  } finally {
    set_hydrating(was_hydrating);
  }
}
const document_listeners = /* @__PURE__ */ new Map();
function _mount(Component, { target, anchor, props = {}, events, context, intro = true }) {
  init_operations();
  var registered_events = /* @__PURE__ */ new Set();
  var event_handle = (events2) => {
    for (var i = 0; i < events2.length; i++) {
      var event_name = events2[i];
      if (registered_events.has(event_name)) continue;
      registered_events.add(event_name);
      var passive = PassiveDelegatedEvents.includes(event_name);
      target.addEventListener(event_name, handle_event_propagation, { passive });
      var n = document_listeners.get(event_name);
      if (n === void 0) {
        document.addEventListener(event_name, handle_event_propagation, { passive });
        document_listeners.set(event_name, 1);
      } else {
        document_listeners.set(event_name, n + 1);
      }
    }
  };
  event_handle(array_from(all_registered_events));
  root_event_handles.add(event_handle);
  var component = void 0;
  var unmount2 = effect_root(() => {
    branch(() => {
      if (context) {
        push({});
        var ctx = (
          /** @type {import('#client').ComponentContext} */
          current_component_context
        );
        ctx.c = context;
      }
      if (events) {
        props.$$events = events;
      }
      if (hydrating) {
        assign_nodes(
          /** @type {import('#client').TemplateNode} */
          anchor,
          null
        );
      }
      component = Component(anchor, props) || {};
      if (hydrating) {
        current_effect.nodes.end = hydrate_node;
      }
      if (context) {
        pop();
      }
    });
    return () => {
      for (var event_name of registered_events) {
        target.removeEventListener(event_name, handle_event_propagation);
        var n = (
          /** @type {number} */
          document_listeners.get(event_name)
        );
        if (--n === 0) {
          document.removeEventListener(event_name, handle_event_propagation);
          document_listeners.delete(event_name);
        } else {
          document_listeners.set(event_name, n);
        }
      }
      root_event_handles.delete(event_handle);
      mounted_components.delete(component);
    };
  });
  mounted_components.set(component, unmount2);
  return component;
}
let mounted_components = /* @__PURE__ */ new WeakMap();
function unmount(component) {
  const fn = mounted_components.get(component);
  fn?.();
}
function asClassComponent$1(component) {
  return class extends Svelte4Component {
    /** @param {any} options */
    constructor(options2) {
      super({
        component,
        ...options2
      });
    }
  };
}
class Svelte4Component {
  /** @type {any} */
  #events;
  /** @type {Record<string, any>} */
  #instance;
  /**
   * @param {ComponentConstructorOptions & {
   *  component: any;
   * 	immutable?: boolean;
   * 	hydrate?: boolean;
   * 	recover?: false;
   * }} options
   */
  constructor(options2) {
    var sources = /* @__PURE__ */ new Map();
    var add_source = (key, value) => {
      var s = /* @__PURE__ */ mutable_source(value);
      sources.set(key, s);
      return s;
    };
    const props = new Proxy(
      { ...options2.props || {}, $$events: {} },
      {
        get(target, prop) {
          return get(sources.get(prop) ?? add_source(prop, Reflect.get(target, prop)));
        },
        has(target, prop) {
          get(sources.get(prop) ?? add_source(prop, Reflect.get(target, prop)));
          return Reflect.has(target, prop);
        },
        set(target, prop, value) {
          set(sources.get(prop) ?? add_source(prop, value), value);
          return Reflect.set(target, prop, value);
        }
      }
    );
    this.#instance = (options2.hydrate ? hydrate : mount)(options2.component, {
      target: options2.target,
      props,
      context: options2.context,
      intro: options2.intro ?? false,
      recover: options2.recover
    });
    this.#events = props.$$events;
    for (const key of Object.keys(this.#instance)) {
      if (key === "$set" || key === "$destroy" || key === "$on") continue;
      define_property(this, key, {
        get() {
          return this.#instance[key];
        },
        /** @param {any} value */
        set(value) {
          this.#instance[key] = value;
        },
        enumerable: true
      });
    }
    this.#instance.$set = /** @param {Record<string, any>} next */
    (next) => {
      Object.assign(props, next);
    };
    this.#instance.$destroy = () => {
      unmount(this.#instance);
    };
  }
  /** @param {Record<string, any>} props */
  $set(props) {
    this.#instance.$set(props);
  }
  /**
   * @param {string} event
   * @param {(...args: any[]) => any} callback
   * @returns {any}
   */
  $on(event, callback) {
    this.#events[event] = this.#events[event] || [];
    const cb = (...args) => callback.call(this, ...args);
    this.#events[event].push(cb);
    return () => {
      this.#events[event] = this.#events[event].filter(
        /** @param {any} fn */
        (fn) => fn !== cb
      );
    };
  }
  $destroy() {
    this.#instance.$destroy();
  }
}
function asClassComponent(component) {
  const component_constructor = asClassComponent$1(component);
  const _render = (props, { context } = {}) => {
    const result = render(component, { props, context });
    return {
      css: { code: "", map: null },
      head: result.head,
      html: result.body
    };
  };
  component_constructor.render = _render;
  return component_constructor;
}
let prerendering = false;
function set_building() {
}
function set_prerendering() {
  prerendering = true;
}
function Root($$payload, $$props) {
  push$1();
  let {
    stores,
    page,
    constructors,
    components = [],
    form,
    data_0 = null,
    data_1 = null
  } = $$props;
  {
    setContext("__svelte__", stores);
  }
  {
    stores.page.set(page);
  }
  if (constructors[1]) {
    $$payload.out += "<!--[-->";
    $$payload.out += `<!---->`;
    constructors[0]?.($$payload, {
      data: data_0,
      children: ($$payload2, $$slotProps) => {
        $$payload2.out += `<!---->`;
        constructors[1]?.($$payload2, { data: data_1, form });
        $$payload2.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload.out += `<!---->`;
  } else {
    $$payload.out += "<!--[!-->";
    $$payload.out += `<!---->`;
    constructors[0]?.($$payload, { data: data_0, form });
    $$payload.out += `<!---->`;
  }
  $$payload.out += `<!--]--> `;
  {
    $$payload.out += "<!--[!-->";
  }
  $$payload.out += `<!--]-->`;
  pop$1();
}
const root = asClassComponent(Root);
function set_read_implementation(fn) {
}
function set_manifest(_) {
}
const options = {
  app_dir: "_app",
  app_template_contains_nonce: false,
  csp: { "mode": "auto", "directives": { "upgrade-insecure-requests": false, "block-all-mixed-content": false }, "reportOnly": { "upgrade-insecure-requests": false, "block-all-mixed-content": false } },
  csrf_check_origin: true,
  embedded: false,
  env_public_prefix: "PUBLIC_",
  env_private_prefix: "",
  hooks: null,
  // added lazily, via `get_hooks`
  preload_strategy: "modulepreload",
  root,
  service_worker: false,
  templates: {
    app: ({ head, body, assets: assets2, nonce, env }) => '<!DOCTYPE html>\n<html lang="en">\n	<head>\n		<meta charset="utf-8" />\n		<meta name="viewport" content="width=device-width, initial-scale=1" />\n		<link rel="icon" href="https://fav.farm/🪄" />\n		' + head + '\n	</head>\n	<body	data-sveltekit-preload-data="hover">\n		<div style="display: contents">' + body + "</div>\n	</body>\n</html>\n",
    error: ({ status, message }) => '<!doctype html>\n<html lang="en">\n	<head>\n		<meta charset="utf-8" />\n		<title>' + message + `</title>

		<style>
			body {
				--bg: white;
				--fg: #222;
				--divider: #ccc;
				background: var(--bg);
				color: var(--fg);
				font-family:
					system-ui,
					-apple-system,
					BlinkMacSystemFont,
					'Segoe UI',
					Roboto,
					Oxygen,
					Ubuntu,
					Cantarell,
					'Open Sans',
					'Helvetica Neue',
					sans-serif;
				display: flex;
				align-items: center;
				justify-content: center;
				height: 100vh;
				margin: 0;
			}

			.error {
				display: flex;
				align-items: center;
				max-width: 32rem;
				margin: 0 1rem;
			}

			.status {
				font-weight: 200;
				font-size: 3rem;
				line-height: 1;
				position: relative;
				top: -0.05rem;
			}

			.message {
				border-left: 1px solid var(--divider);
				padding: 0 0 0 1rem;
				margin: 0 0 0 1rem;
				min-height: 2.5rem;
				display: flex;
				align-items: center;
			}

			.message h1 {
				font-weight: 400;
				font-size: 1em;
				margin: 0;
			}

			@media (prefers-color-scheme: dark) {
				body {
					--bg: #222;
					--fg: #ddd;
					--divider: #666;
				}
			}
		</style>
	</head>
	<body>
		<div class="error">
			<span class="status">` + status + '</span>\n			<div class="message">\n				<h1>' + message + "</h1>\n			</div>\n		</div>\n	</body>\n</html>\n"
  },
  version_hash: "ha2ows"
};
async function get_hooks() {
  return {};
}
export {
  assets as a,
  base as b,
  safe_public_env as c,
  options as d,
  set_private_env as e,
  prerendering as f,
  set_public_env as g,
  get_hooks as h,
  set_safe_public_env as i,
  set_assets as j,
  set_building as k,
  set_manifest as l,
  set_prerendering as m,
  set_read_implementation as n,
  override as o,
  public_env as p,
  reset as r,
  safe_not_equal as s
};
