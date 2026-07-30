use std::cell::RefCell;
use std::mem;
use std::slice;

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::ApproximateTokEnv;
use llguidance::{Matcher, ParserFactory};

const MAX_GRAMMAR_BYTES: usize = 256 * 1024;
const MAX_INPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_MATCHERS: usize = 4096;

const ERR_INVALID_INPUT: i32 = -1;
const ERR_INVALID_GRAMMAR: i32 = -2;
const ERR_RESOURCE_LIMIT: i32 = -3;
const ERR_INTERNAL: i32 = -4;

thread_local! {
    static FACTORY: RefCell<Option<ParserFactory>> = const { RefCell::new(None) };
    static MATCHERS: RefCell<Vec<Option<Matcher>>> = const { RefCell::new(Vec::new()) };
}

#[no_mangle]
pub extern "C" fn axon_alloc(size: u32) -> u32 {
    if size == 0 {
        return 0;
    }
    let mut bytes = Vec::<u8>::with_capacity(size as usize);
    let pointer = bytes.as_mut_ptr();
    mem::forget(bytes);
    pointer as u32
}

#[no_mangle]
pub unsafe extern "C" fn axon_dealloc(pointer: u32, capacity: u32) {
    if pointer == 0 || capacity == 0 {
        return;
    }
    drop(Vec::from_raw_parts(
        pointer as *mut u8,
        0,
        capacity as usize,
    ));
}

#[no_mangle]
pub unsafe extern "C" fn axon_compile(
    syntax_pointer: u32,
    syntax_length: u32,
    grammar_pointer: u32,
    grammar_length: u32,
) -> i32 {
    if grammar_length as usize > MAX_GRAMMAR_BYTES {
        return ERR_RESOURCE_LIMIT;
    }
    let Some(syntax_bytes) = borrowed_bytes(syntax_pointer, syntax_length) else {
        return ERR_INVALID_INPUT;
    };
    let Some(grammar_bytes) = borrowed_bytes(grammar_pointer, grammar_length) else {
        return ERR_INVALID_INPUT;
    };
    let Ok(syntax) = std::str::from_utf8(syntax_bytes) else {
        return ERR_INVALID_INPUT;
    };
    let Ok(grammar) = std::str::from_utf8(grammar_bytes) else {
        return ERR_INVALID_INPUT;
    };
    if syntax != "lark" && syntax != "regex" {
        return ERR_INVALID_INPUT;
    }

    let Ok(top_level) = TopLevelGrammar::from_tagged_str(syntax, grammar) else {
        return ERR_INVALID_GRAMMAR;
    };
    let matcher = FACTORY.with(|factory| {
        let mut factory = factory.borrow_mut();
        if factory.is_none() {
            let token_environment = ApproximateTokEnv::single_byte_env();
            let Ok(mut created) = ParserFactory::new_simple(&token_environment) else {
                return Err(ERR_INTERNAL);
            };
            created.quiet();
            *factory = Some(created);
        }
        let Some(factory) = factory.as_ref() else {
            return Err(ERR_INTERNAL);
        };
        Ok(Matcher::new(factory.create_parser(top_level)))
    });
    let Ok(matcher) = matcher else {
        return ERR_INTERNAL;
    };
    if matcher.is_error() {
        return ERR_INVALID_GRAMMAR;
    }

    MATCHERS.with(|matchers| {
        let mut matchers = matchers.borrow_mut();
        if let Some(index) = matchers.iter().position(Option::is_none) {
            matchers[index] = Some(matcher);
            return (index + 1) as i32;
        }
        if matchers.len() >= MAX_MATCHERS {
            return ERR_RESOURCE_LIMIT;
        }
        matchers.push(Some(matcher));
        matchers.len() as i32
    })
}

#[no_mangle]
pub unsafe extern "C" fn axon_validate(
    handle: u32,
    input_pointer: u32,
    input_length: u32,
) -> i32 {
    if handle == 0 {
        return ERR_INVALID_INPUT;
    }
    if input_length as usize > MAX_INPUT_BYTES {
        return ERR_RESOURCE_LIMIT;
    }
    let Some(input) = borrowed_bytes(input_pointer, input_length) else {
        return ERR_INVALID_INPUT;
    };

    MATCHERS.with(|matchers| {
        let matchers = matchers.borrow();
        let Some(template) = matchers
            .get(handle as usize - 1)
            .and_then(Option::as_ref)
        else {
            return ERR_INVALID_INPUT;
        };
        let mut matcher = template.deep_clone();
        let tokens = input.iter().map(|byte| *byte as u32).collect::<Vec<_>>();
        match matcher.try_consume_tokens(&tokens) {
            Ok(consumed) if consumed == tokens.len() => match matcher.is_accepting() {
                Ok(true) => 1,
                Ok(false) => 0,
                Err(_) => ERR_INTERNAL,
            },
            Ok(_) => 0,
            Err(_) => ERR_INTERNAL,
        }
    })
}

#[no_mangle]
pub extern "C" fn axon_free(handle: u32) -> i32 {
    if handle == 0 {
        return ERR_INVALID_INPUT;
    }
    MATCHERS.with(|matchers| {
        let mut matchers = matchers.borrow_mut();
        let Some(slot) = matchers.get_mut(handle as usize - 1) else {
            return ERR_INVALID_INPUT;
        };
        if slot.take().is_none() {
            return ERR_INVALID_INPUT;
        }
        0
    })
}

unsafe fn borrowed_bytes(pointer: u32, length: u32) -> Option<&'static [u8]> {
    if length == 0 {
        return Some(&[]);
    }
    if pointer == 0 {
        return None;
    }
    Some(slice::from_raw_parts(
        pointer as *const u8,
        length as usize,
    ))
}
