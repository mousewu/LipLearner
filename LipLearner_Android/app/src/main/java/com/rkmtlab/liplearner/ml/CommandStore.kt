package com.rkmtlab.liplearner.ml

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.sqrt

/**
 * Holds all learned state and persists it. Android replacement for the iOS combination of
 * `trainDataFrame` (CSV), `KWSDataFrame` (CSV) and `commandCenterDict` (NSKeyedArchiver .dat).
 *
 * Everything is serialized to a single JSON file per user under the app's files dir.
 */
class CommandStore(
    private val context: Context,
    private val embedDim: Int,
    private val modelId: String,
) {

    class Center(var count: Float, val mean: FloatArray) {
        fun add(vec: FloatArray) {
            if (count == 0f) {
                System.arraycopy(vec, 0, mean, 0, vec.size)
                count = 1f
            } else {
                for (i in mean.indices) mean[i] = (mean[i] * count + vec[i]) / (count + 1f)
                count += 1f
            }
        }
    }

    data class Sample(val label: String, val vector: FloatArray)

    val commandSamples = mutableListOf<Sample>()          // -> command classifier
    val kwsSamples = mutableListOf<Sample>()              // labels "P"/"N" -> KWS classifier
    val commandCenters = linkedMapOf<String, Center>()    // for similarity sort
    var keywordCenter = Center(0f, FloatArray(embedDim))
    var nonSpeakingCenter = Center(0f, FloatArray(embedDim))

    @Volatile var commandClassifier: SoftmaxRegression? = null
    @Volatile var kwsClassifier: SoftmaxRegression? = null

    val registeredCommands: List<String> get() = commandCenters.keys.sorted()

    // --- mutations -----------------------------------------------------------

    fun addCommandSample(label: String, vector: FloatArray) {
        commandSamples.add(Sample(label, vector))
        commandCenters.getOrPut(label) { Center(0f, FloatArray(embedDim)) }.add(vector)
    }

    fun addKwsSample(label: String, vector: FloatArray) {
        kwsSamples.add(Sample(label, vector))
    }

    fun addKeywordSample(vector: FloatArray) = keywordCenter.add(vector)
    fun addNonSpeakingSample(vector: FloatArray) = nonSpeakingCenter.add(vector)

    fun resetKeyword() {
        keywordCenter = Center(0f, FloatArray(embedDim))
        nonSpeakingCenter = Center(0f, FloatArray(embedDim))
        kwsClassifier = null
    }

    val kwsReady: Boolean get() = keywordCenter.count > 0 && nonSpeakingCenter.count > 0

    // --- training ------------------------------------------------------------

    fun trainAll() {
        commandClassifier = SoftmaxRegression.train(commandSamples.map { it.label to it.vector })
        kwsClassifier = SoftmaxRegression.train(kwsSamples.map { it.label to it.vector })
    }

    fun trainCommand() {
        commandClassifier = SoftmaxRegression.train(commandSamples.map { it.label to it.vector })
    }

    fun trainKws() {
        kwsClassifier = SoftmaxRegression.train(kwsSamples.map { it.label to it.vector })
    }

    // --- similarity ranking (iOS sortCommandsBySimilarity) -------------------

    /**
     * Ranks registered commands by score = dot(query, center) * ||center||, ascending — replicating
     * the exact ordering used by the iOS picker in the recognition-confirm dialog.
     */
    fun sortCommandsBySimilarity(query: FloatArray): List<String> {
        return commandCenters.entries
            .map { (name, center) ->
                var dot = 0f
                var sumSq = 0f
                for (i in query.indices) {
                    dot += query[i] * center.mean[i]
                    sumSq += center.mean[i] * center.mean[i]
                }
                name to dot * sqrt(sumSq)
            }
            .sortedBy { it.second }
            .map { it.first }
    }

    // --- persistence ---------------------------------------------------------

    // Per-model file: embeddings from different models live in different spaces, so each model keeps
    // its own commands / keyword data.
    private fun file(userName: String) = File(context.filesDir, "${userName}__${modelId}.json")

    fun save(userName: String) {
        val root = JSONObject()
        root.put("commandSamples", samplesToJson(commandSamples))
        root.put("kwsSamples", samplesToJson(kwsSamples))
        val centers = JSONObject()
        for ((name, c) in commandCenters) centers.put(name, centerToJson(c))
        root.put("commandCenters", centers)
        root.put("keywordCenter", centerToJson(keywordCenter))
        root.put("nonSpeakingCenter", centerToJson(nonSpeakingCenter))
        file(userName).writeText(root.toString())
    }

    fun load(userName: String): Boolean {
        val f = file(userName)
        if (!f.exists()) return false
        val root = JSONObject(f.readText())
        commandSamples.clear(); commandSamples.addAll(samplesFromJson(root.getJSONArray("commandSamples")))
        kwsSamples.clear(); kwsSamples.addAll(samplesFromJson(root.getJSONArray("kwsSamples")))
        commandCenters.clear()
        val centers = root.getJSONObject("commandCenters")
        for (name in centers.keys()) commandCenters[name] = centerFromJson(centers.getJSONObject(name))
        keywordCenter = centerFromJson(root.getJSONObject("keywordCenter"))
        nonSpeakingCenter = centerFromJson(root.getJSONObject("nonSpeakingCenter"))
        return true
    }

    private fun samplesToJson(list: List<Sample>): JSONArray {
        val arr = JSONArray()
        for (s in list) {
            val o = JSONObject()
            o.put("label", s.label)
            o.put("vector", floatsToJson(s.vector))
            arr.put(o)
        }
        return arr
    }

    private fun samplesFromJson(arr: JSONArray): List<Sample> = buildList {
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            add(Sample(o.getString("label"), floatsFromJson(o.getJSONArray("vector"))))
        }
    }

    private fun centerToJson(c: Center) = JSONObject().apply {
        put("count", c.count.toDouble())
        put("mean", floatsToJson(c.mean))
    }

    private fun centerFromJson(o: JSONObject) =
        Center(o.getDouble("count").toFloat(), floatsFromJson(o.getJSONArray("mean")))

    private fun floatsToJson(v: FloatArray): JSONArray {
        val a = JSONArray()
        for (x in v) a.put(x.toDouble())
        return a
    }

    private fun floatsFromJson(a: JSONArray): FloatArray =
        FloatArray(a.length()) { a.getDouble(it).toFloat() }
}
