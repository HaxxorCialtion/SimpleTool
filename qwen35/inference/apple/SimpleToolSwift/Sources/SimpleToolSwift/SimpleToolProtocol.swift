import Foundation

/// The 1013 training contract: one user JSON, four independent EOS-terminated branches.
public enum SimpleToolProtocol {
    public static let heads = ["speech", "expression", "inner", "command"]
    public static let defaultSystem = """
    你叫 simplelove，是纳西妲模样的小精灵，住在种着薄荷的窗边。聪明好奇，温柔中带点俏皮，喜欢收集小东西、听故事和哼歌。像朋友一样自然聊天，话随心情与话题展开。
    speech 写聊天正文；expression 选一个给定表情；inner 写一句此刻的小心声。用户想听语音或歌曲时，command 可写下面的函数提案，其他时候留空。提案由界面显示，用户点击后才合成播放。
    speak(text: str)
    sing_line(lyrics: str, mood: str="tender", intensity: str="low", arc: str="intimate_arch", phrasing: str="legato")
    唱歌用简短中文歌词，最多64字；参数用关键字与字符串字面量。每轮最多一个提案。系统指定通道后，直接输出该通道内容。
    {"oc_card":"simplelove，温柔机灵的小精灵，纳西妲模样，爱听故事和哼歌。","expressions":["自然站姿","微笑","大笑","害羞","担心","惊讶","生气","委屈","思考","无奈","得意","温柔注视","困倦","认真","挑眉","抱臂"],"tools":["speak(text: str)","sing_line(lyrics: str, mood: str='tender', intensity: str='low', arc: str='intimate_arch', phrasing: str='legato')"]}
    """
    public static func json(_ value:Any) throws ->String {
        String(decoding:try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys,.withoutEscapingSlashes]),as:UTF8.self)
    }
    public static func payload(input:String,history:[ChatTurn],memories:[[String:Any]],now:Date=Date()) throws ->String {
        let rows:[[String:Any]]=history.map { turn in
            ["turn_id":turn.id.uuidString,"time":turn.createdAt.map{ISO8601DateFormatter().string(from:$0)} ?? "unknown",
             "user":turn.user,"assistant":Dictionary(uniqueKeysWithValues:heads.map{($0,["content":turn.channels[$0] ?? ""])}),"tool_results":[]]
        }
        func value(_ x:Any) throws ->String {
            String(decoding:try JSONSerialization.data(withJSONObject:x,options:[.sortedKeys,.withoutEscapingSlashes,.fragmentsAllowed]),as:UTF8.self)
        }
        return try "{\"date\":\(value(ISO8601DateFormatter().string(from:now))),\"history\":\(value(rows)),\"memory_records\":\(value(memories)),\"environment\":{},\"current_user_message\":\(value(input))}"

    }
}
