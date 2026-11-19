import Foundation

/// Text-only implementation of the actual checkpoint's NFC + Split + ByteLevel BPE.
/// Model markers and EOS are read from the vocabulary; no fixed token IDs.
public final class ChatTokenizer {
    private let vocabulary:[String:Int],ranks:[String:Int],bytes:[UInt8:String]
    private let specials:[String:Int],specialPattern:NSRegularExpression,split:NSRegularExpression
    private var memo=[String:[Int]]()
    public init(directory:URL) throws {
        let j=try JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("tokenizer.json"))) as! [String:Any]
        guard let m=j["model"] as? [String:Any],m["type"] as? String == "BPE",
              let v=m["vocab"] as? [String:Int],let merges=m["merges"] as? [[String]],
              let normalizer=j["normalizer"] as? [String:Any],normalizer["type"] as? String == "NFC",
              let pre=j["pre_tokenizer"] as? [String:Any],let stages=pre["pretokenizers"] as? [[String:Any]],
              stages.count==2,let pat=stages[0]["pattern"] as? [String:String],let regex=pat["Regex"],
              stages[1]["type"] as? String == "ByteLevel",stages[1]["add_prefix_space"] as? Bool == false
        else {throw EngineError.invalidConfiguration("Unsupported tokenizer pipeline")}
        vocabulary=v
        ranks=Dictionary(uniqueKeysWithValues:merges.enumerated().map{($0.element.joined(separator:"\0"),$0.offset)})
        let visible=Array(33...126)+Array(161...172)+Array(174...255)
        var map=[UInt8:String](),extra=256
        for b in visible {map[UInt8(b)]=String(UnicodeScalar(b)!)}
        for b in 0...255 where map[UInt8(b)]==nil {map[UInt8(b)]=String(UnicodeScalar(extra)!);extra+=1}
        bytes=map
        specials=Dictionary(uniqueKeysWithValues:(j["added_tokens"] as? [[String:Any]] ?? []).compactMap {
            guard let s=$0["content"] as? String,let id=$0["id"] as? Int else {return nil}
            return (s,id)
        })
        let alternation=specials.keys.sorted{$0.count>$1.count}.map{NSRegularExpression.escapedPattern(for:$0)}.joined(separator:"|")
        specialPattern=try NSRegularExpression(pattern:alternation)
        split=try NSRegularExpression(pattern:regex)
    }
    public func tokenID(_ text:String) throws ->Int {
        guard let id=specials[text] ?? vocabulary[text] else {throw EngineError.invalidPacket("Missing token \(text)")}
        return id
    }
    public func encode(_ text:String) throws ->[Int] {
        // Drain Objective-C regex/NSString temporaries after each tokenization.
        // Only the native Swift token array escapes this pool.
        try autoreleasepool { try encodeUnpooled(text) }
    }
    private func encodeUnpooled(_ text:String) throws ->[Int] {
        let s=text as NSString
        var result=[Int](),offset=0
        for match in specialPattern.matches(in:text,range:NSRange(location:0,length:s.length)) {
            if match.range.location>offset {result += try ordinary(s.substring(with:NSRange(location:offset,length:match.range.location-offset)))}
            result.append(specials[s.substring(with:match.range)]!)
            offset=NSMaxRange(match.range)
        }
        if offset<s.length {result += try ordinary(s.substring(from:offset))}
        return result
    }
    private func ordinary(_ text:String) throws ->[Int] {
        let normalized=text.precomposedStringWithCanonicalMapping,s=normalized as NSString
        var result=[Int](),cursor=0
        for match in split.matches(in:normalized,range:NSRange(location:0,length:s.length)) {
            guard match.range.location==cursor else {throw EngineError.invalidPacket("Tokenizer split left a gap")}
            cursor=NSMaxRange(match.range)
            result += try bpe(s.substring(with:match.range))
        }
        guard cursor==s.length else {throw EngineError.invalidPacket("Tokenizer split missed suffix")}
        return result
    }
    private func bpe(_ text:String) throws ->[Int] {
        if let hit=memo[text] {return hit}
        var parts=text.utf8.map{bytes[$0]!}
        while parts.count>1 {
            var rank=Int.max,pair:String?
            for i in 0..<(parts.count-1) {
                let key=parts[i]+"\0"+parts[i+1]
                if let r=ranks[key],r<rank {rank=r;pair=key}
            }
            guard let pair else {break}
            var next=[String](),i=0
            while i<parts.count {
                if i+1<parts.count,parts[i]+"\0"+parts[i+1]==pair {next.append(parts[i]+parts[i+1]);i+=2}
                else {next.append(parts[i]);i+=1}
            }
            parts=next
        }
        let ids=try parts.map {piece ->Int in
            guard let id=vocabulary[piece] else {throw EngineError.invalidPacket("Missing BPE piece")}
            return id
        }
        if memo.count<4096 {memo[text]=ids}
        return ids
    }
    public static let generationSuffix="<|im_start|>assistant\n<think>\n\n</think>\n\n"
    public static func message(_ role:String,_ content:String)->String {
        "<|im_start|>\(role)\n\(content.trimmingCharacters(in:.whitespacesAndNewlines))<|im_end|>\n"
    }
}
