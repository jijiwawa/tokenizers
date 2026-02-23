//! ARM NEON指令优化的字符串处理工具函数
//! 仅在ARM64架构上启用

/// 检查字符串中是否包含指定的特殊字符
/// 使用ARM NEON指令集进行并行处理，提高性能
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn contains_special_characters(s: &str, special_chars: &[char]) -> bool {
    use std::arch::aarch64::*;
    
    // 将特殊字符转换为字节进行比较
    let special_bytes: Vec<u8> = special_chars.iter()
        .filter_map(|c| if c.is_ascii() { Some(*c as u8) } else { None })
        .collect();
    
    if special_bytes.is_empty() {
        return false;
    }
    
    let bytes = s.as_bytes();
    let len = bytes.len();
    
    // 如果字符串太短，使用常规方法
    if len < 16 {
        return bytes.iter().any(|&b| special_bytes.contains(&b));
    }
    
    // 准备NEON寄存器
    unsafe {
        let mut i = 0;
        
        // 批量处理16字节数据
        while i + 16 <= len {
            // 加载16字节数据
            let data = vld1q_u8(&bytes[i]);
            
            // 对每个特殊字符进行比较
            for &c in special_bytes.iter() {
                let char_vec = vdupq_n_u8(c);
                let result = vceqq_u8(data, char_vec);
                
                // 检查是否有匹配
                if vmaxvq_u8(result) != 0 {
                    return true;
                }
            }
            
            i += 16;
        }
        
        // 处理剩余的字节
        for &b in bytes[i..].iter() {
            if special_bytes.contains(&b) {
                return true;
            }
        }
    }
    
    false
}

/// 非ARM64架构的回退实现
#[cfg(not(target_arch = "aarch64"))]
pub fn contains_special_characters(s: &str, special_chars: &[char]) -> bool {
    let special_bytes: Vec<u8> = special_chars.iter()
        .filter_map(|c| if c.is_ascii() { Some(*c as u8) } else { None })
        .collect();
    
    s.as_bytes().iter().any(|&b| special_bytes.contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_contains_special_characters() {
        let special_chars = ['<', '>', '|', '[', ']'];
        
        assert!(contains_special_characters("Hello <world>", &special_chars));
        assert!(!contains_special_characters("Hello world", &special_chars));
        assert!(contains_special_characters("<", &special_chars));
        assert!(!contains_special_characters("", &special_chars));
        
        // 测试较长的字符串
        let long_str = "a".repeat(20) + "|" + &"a".repeat(20);
        assert!(contains_special_characters(&long_str, &special_chars));
    }
}