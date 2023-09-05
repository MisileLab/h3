use dioxus::prelude::*;

fn main() {
    // launch the dioxus app in a webview
    dioxus_desktop::launch(app);
}

// define a component that renders a div with the text "Hello, world!"
fn app(cx: Scope) -> Element {
    cx.render(rsx! {
        div {
            h1 { "Hello, World!" }
        }
    })
}
