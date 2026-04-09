//! ARM NEON指令优化的字符串处理工具函数
//! 仅在ARM64架构上启用

/// 检查字符串中是否包含任意一个 ASCII 字节。
/// 使用 ARM NEON 指令集对长字符串做批量扫描，避免热路径上的额外分配。
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn contains_any_ascii_byte(s: &str, ascii_bytes: &[u8]) -> bool {
    use std::arch::aarch64::*;

    if ascii_bytes.is_empty() {
        return false;
    }

    let bytes = s.as_bytes();
    if bytes.len() < 16 {
        return bytes
            .iter()
            .any(|&byte| ascii_bytes.iter().any(|&needle| needle == byte));
    }

    unsafe {
        let mut i = 0;

        while i + 16 <= bytes.len() {
            let data = vld1q_u8(bytes.as_ptr().add(i));
            let mut matches = vdupq_n_u8(0);

            for &needle in ascii_bytes {
                let mask = vceqq_u8(data, vdupq_n_u8(needle));
                matches = vorrq_u8(matches, mask);
            }

            if vmaxvq_u8(matches) != 0 {
                return true;
            }

            i += 16;
        }

        for &byte in &bytes[i..] {
            if ascii_bytes.iter().any(|&needle| needle == byte) {
                return true;
            }
        }
    }

    false
}

/// 非 ARM64 架构的回退实现。
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn contains_any_ascii_byte(s: &str, ascii_bytes: &[u8]) -> bool {
    s.as_bytes()
        .iter()
        .any(|&byte| ascii_bytes.iter().any(|&needle| needle == byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains_any_ascii_byte() {
        let ascii_bytes = [b'<', b'>', b'|', b'[', b']'];

        assert!(contains_any_ascii_byte("Hello <world>", &ascii_bytes));
        assert!(!contains_any_ascii_byte("Hello world", &ascii_bytes));
        assert!(contains_any_ascii_byte("<", &ascii_bytes));
        assert!(!contains_any_ascii_byte("", &ascii_bytes));

        let long_str = "a".repeat(20) + "|" + &"a".repeat(20);
        assert!(contains_any_ascii_byte(&long_str, &ascii_bytes));
    }
}
