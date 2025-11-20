use crossbeam::channel::{unbounded, Sender, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::collections::HashMap;

// Actor system for Flux
// Provides message-passing based concurrency with isolated state

pub struct Actor<T> {
    sender: Sender<Message<T>>,
    handle: Option<thread::JoinHandle<()>>,
}

pub enum Message<T> {
    User(T),
    Stop,
}

pub struct ActorContext<T> {
    #[allow(dead_code)]
    receiver: Receiver<Message<T>>,
    #[allow(dead_code)]
    state: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl<T: Send + 'static> Actor<T> {
    pub fn spawn<F>(initial_state: T, handler: F) -> Self
    where
        F: Fn(&mut T, T) + Send + 'static,
        T: Clone,
    {
        let (sender, receiver) = unbounded();

        let handle = thread::spawn(move || {
            let mut state = initial_state;

            loop {
                match receiver.recv() {
                    Ok(Message::User(msg)) => {
                        handler(&mut state, msg);
                    }
                    Ok(Message::Stop) | Err(_) => {
                        break;
                    }
                }
            }
        });

        Actor {
            sender,
            handle: Some(handle),
        }
    }

    pub fn send(&self, msg: T) -> Result<(), String> {
        self.sender
            .send(Message::User(msg))
            .map_err(|e| format!("Failed to send message: {}", e))
    }

    pub fn stop(mut self) {
        let _ = self.sender.send(Message::Stop);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl<T> Drop for Actor<T> {
    fn drop(&mut self) {
        let _ = self.sender.send(Message::Stop);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// Actor system supervisor
pub struct ActorSystem {
    actors: Arc<Mutex<HashMap<String, Box<dyn std::any::Any + Send>>>>,
}

impl ActorSystem {
    pub fn new() -> Self {
        ActorSystem {
            actors: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn spawn_actor<T, F>(&self, name: String, initial_state: T, handler: F) -> Result<(), String>
    where
        T: Send + Clone + 'static,
        F: Fn(&mut T, T) + Send + 'static,
    {
        let actor = Actor::spawn(initial_state, handler);

        let mut actors = self.actors.lock().unwrap();
        actors.insert(name, Box::new(actor));

        Ok(())
    }

    pub fn send_message<T: Send + 'static>(&self, name: &str, msg: T) -> Result<(), String> {
        let actors = self.actors.lock().unwrap();

        let actor = actors
            .get(name)
            .ok_or_else(|| format!("Actor '{}' not found", name))?;

        if let Some(actor) = actor.downcast_ref::<Actor<T>>() {
            actor.send(msg)
        } else {
            Err(format!("Actor '{}' has wrong type", name))
        }
    }

    pub fn stop_actor(&self, name: &str) {
        let mut actors = self.actors.lock().unwrap();
        actors.remove(name);
    }

    pub fn shutdown(self) {
        let mut actors = self.actors.lock().unwrap();
        actors.clear();
    }
}

impl Default for ActorSystem {
    fn default() -> Self {
        Self::new()
    }
}

// Mailbox for actor message queuing
pub struct Mailbox<T> {
    messages: Receiver<T>,
    sender: Sender<T>,
}

impl<T> Mailbox<T> {
    pub fn new() -> Self {
        let (sender, receiver) = unbounded();
        Mailbox {
            messages: receiver,
            sender,
        }
    }

    pub fn send(&self, msg: T) -> Result<(), String> {
        self.sender
            .send(msg)
            .map_err(|e| format!("Failed to send: {}", e))
    }

    pub fn receive(&self) -> Option<T> {
        self.messages.recv().ok()
    }

    pub fn try_receive(&self) -> Option<T> {
        self.messages.try_recv().ok()
    }

    pub fn get_sender(&self) -> Sender<T> {
        self.sender.clone()
    }
}

impl<T> Default for Mailbox<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Supervised actor with restart capability
pub struct SupervisedActor<T> {
    name: String,
    restarts: usize,
    max_restarts: usize,
    actor: Option<Actor<T>>,
}

impl<T: Send + Clone + 'static> SupervisedActor<T> {
    pub fn new<F>(name: String, max_restarts: usize, initial_state: T, handler: F) -> Self
    where
        F: Fn(&mut T, T) + Send + Clone + 'static,
    {
        let actor = Actor::spawn(initial_state.clone(), handler.clone());

        SupervisedActor {
            name,
            restarts: 0,
            max_restarts,
            actor: Some(actor),
        }
    }

    pub fn send(&self, msg: T) -> Result<(), String> {
        if let Some(actor) = &self.actor {
            actor.send(msg)
        } else {
            Err("Actor is not running".to_string())
        }
    }

    pub fn restart<F>(&mut self, initial_state: T, handler: F) -> Result<(), String>
    where
        F: Fn(&mut T, T) + Send + 'static,
    {
        if self.restarts >= self.max_restarts {
            return Err(format!(
                "Actor '{}' exceeded max restarts ({})",
                self.name, self.max_restarts
            ));
        }

        self.actor = None;
        self.actor = Some(Actor::spawn(initial_state, handler));
        self.restarts += 1;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_actor_basic() {
        use std::sync::atomic::{AtomicI32, Ordering};

        let counter = Arc::new(AtomicI32::new(0));
        let counter_clone = counter.clone();

        let actor = Actor::spawn(0, move |state, msg: i32| {
            *state += msg;
            counter_clone.fetch_add(msg, Ordering::SeqCst);
        });

        actor.send(1).unwrap();
        actor.send(2).unwrap();
        actor.send(3).unwrap();

        thread::sleep(Duration::from_millis(100));

        assert_eq!(counter.load(Ordering::SeqCst), 6);

        actor.stop();
    }

    #[test]
    fn test_actor_system() {
        let system = ActorSystem::new();

        system
            .spawn_actor("counter".to_string(), 0, |state, msg: i32| {
                *state += msg;
            })
            .unwrap();

        system.send_message("counter", 5).unwrap();
        system.send_message("counter", 10).unwrap();

        thread::sleep(Duration::from_millis(100));

        system.stop_actor("counter");
    }

    #[test]
    fn test_mailbox() {
        let mailbox = Mailbox::new();

        mailbox.send(1).unwrap();
        mailbox.send(2).unwrap();
        mailbox.send(3).unwrap();

        assert_eq!(mailbox.receive(), Some(1));
        assert_eq!(mailbox.receive(), Some(2));
        assert_eq!(mailbox.receive(), Some(3));
    }
}
