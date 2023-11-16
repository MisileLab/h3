use dioxus::prelude::*;

const REQUEST_TYPE: [&str; 9] = ["GET","HEAD","POST","PUT","DELETE","CONNECT","OPTIONS","TRACE","PATCH"];

fn main() {
    pretty_env_logger::init();
    dioxus_desktop::launch(app);
}

fn app(cx: Scope) -> Element {
    let address = use_state(cx, || "");
    let name = use_state(cx, || "");
    let request_type = use_state(cx, || "");
    cx.render(rsx! {
        div {
            select {
                required: true,
                for i in 0..REQUEST_TYPE.len()-1 {
                    option {
                        REQUEST_TYPE[i]
                    }
                }
            }
        }
    })
}
