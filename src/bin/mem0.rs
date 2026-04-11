use mem0_rust::{AddOptions, Memory, MemoryConfig, SearchOptions};
use std::io::{self, Write};

fn print_help() {
    println!("mem0 CLI (experimental)");
    println!("Commands:");
    println!("  add <user_id> <text>       Add a memory for a user");
    println!("  search <user_id> <query>   Search memories for a user");
    println!("  help                       Show this message");
    println!("  exit                       Quit the shell");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let memory = Memory::new(MemoryConfig::default()).await?;

    println!("Starting mem0 interactive shell.");
    print_help();

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.eq_ignore_ascii_case("exit") || line.eq_ignore_ascii_case("quit") {
            break;
        }

        if line.eq_ignore_ascii_case("help") {
            print_help();
            continue;
        }

        let mut parts = line.splitn(3, ' ');
        let cmd = parts.next().unwrap_or_default();

        match cmd {
            "add" => {
                let user_id = match parts.next() {
                    Some(v) if !v.is_empty() => v,
                    _ => {
                        println!("usage: add <user_id> <text>");
                        continue;
                    }
                };
                let text = match parts.next() {
                    Some(v) if !v.is_empty() => v,
                    _ => {
                        println!("usage: add <user_id> <text>");
                        continue;
                    }
                };

                let result = memory
                    .add(
                        text,
                        AddOptions {
                            user_id: Some(user_id.to_string()),
                            ..Default::default()
                        },
                    )
                    .await;

                match result {
                    Ok(add_result) => println!("added {} event(s)", add_result.results.len()),
                    Err(err) => println!("error: {err}"),
                }
            }
            "search" => {
                let user_id = match parts.next() {
                    Some(v) if !v.is_empty() => v,
                    _ => {
                        println!("usage: search <user_id> <query>");
                        continue;
                    }
                };
                let query = match parts.next() {
                    Some(v) if !v.is_empty() => v,
                    _ => {
                        println!("usage: search <user_id> <query>");
                        continue;
                    }
                };

                let result = memory
                    .search(
                        query,
                        SearchOptions {
                            user_id: Some(user_id.to_string()),
                            limit: Some(5),
                            ..Default::default()
                        },
                    )
                    .await;

                match result {
                    Ok(search_result) => {
                        if search_result.results.is_empty() {
                            println!("no results");
                            continue;
                        }

                        for item in search_result.results {
                            println!("- {:.3}: {}", item.score, item.record.content);
                        }
                    }
                    Err(err) => println!("error: {err}"),
                }
            }
            _ => {
                println!("unknown command: {cmd}");
                print_help();
            }
        }
    }

    println!("bye");
    Ok(())
}
