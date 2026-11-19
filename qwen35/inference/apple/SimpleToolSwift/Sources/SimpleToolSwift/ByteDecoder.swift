import Foundation

// Decode the checkpoint's ByteLevel BPE vocabulary without assuming that a
// closing tag occupies complete token boundaries. Encoding remains external.
final class ByteDecoder {
    private let pieces:[Int:[UInt8]]
    init(_ url:URL) throws {
        let json=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as! [String:Any]
        guard let model=json["model"] as? [String:Any],model["type"] as? String == "BPE",
              let decoder=json["decoder"] as? [String:Any],decoder["type"] as? String == "ByteLevel",
              let vocab=model["vocab"] as? [String:Int] else {
            throw EngineError.invalidConfiguration("requires ByteLevel BPE tokenizer")
        }
        let visible=Array(33...126)+Array(161...172)+Array(174...255)
        var reverse=[UInt32:UInt8](),extra:UInt32=256
        for byte in visible {reverse[UInt32(byte)]=UInt8(byte)}
        for byte in 0...255 where reverse[UInt32(byte)] == nil {
            reverse[extra]=UInt8(byte);extra+=1
        }
        var result=[Int:[UInt8]]()
        for (text,id) in vocab {
            var bytes=[UInt8]()
            for scalar in text.unicodeScalars {
                guard let byte=reverse[scalar.value] else {throw EngineError.invalidConfiguration("invalid ByteLevel symbol")}
                bytes.append(byte)
            }
            result[id]=bytes
        }
        for item in json["added_tokens"] as? [[String:Any]] ?? [] {
            if let id=item["id"] as? Int,let text=item["content"] as? String {result[id]=Array(text.utf8)}
        }
        pieces=result
    }
    func decode(_ ids:[Int]) throws ->String {
        var bytes=[UInt8]()
        for id in ids {
            guard let piece=pieces[id] else {throw EngineError.invalidPacket("unknown token id \(id)")}
            bytes.append(contentsOf:piece)
        }
        return String(decoding:bytes,as:UTF8.self)
    }
}
