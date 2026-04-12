use std::future::Future;
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub attempts: u32,
    pub base_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            attempts: 3,
            base_delay_ms: 150,
        }
    }
}

pub async fn retry_async<T, E, F, Fut>(policy: RetryPolicy, mut f: F) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let attempts = policy.attempts.max(1);
    let mut delay_ms = policy.base_delay_ms;

    for attempt in 1..=attempts {
        match f().await {
            Ok(value) => return Ok(value),
            Err(err) if attempt == attempts => return Err(err),
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                delay_ms = (delay_ms * 2).min(5_000);
            }
        }
    }

    unreachable!("retry loop always returns")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn retries_until_success() {
        let attempts = AtomicUsize::new(0);
        let result = retry_async(
            RetryPolicy {
                attempts: 3,
                base_delay_ms: 1,
            },
            || async {
                let current = attempts.fetch_add(1, Ordering::SeqCst);
                if current < 2 {
                    Err("transient")
                } else {
                    Ok("ok")
                }
            },
        )
        .await;

        assert_eq!(result.unwrap(), "ok");
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }
}
