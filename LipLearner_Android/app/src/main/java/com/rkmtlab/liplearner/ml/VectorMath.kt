package com.rkmtlab.liplearner.ml

import kotlin.math.sqrt

/** Small vector helpers, matching the vDSP calls used in the iOS controller. */
object VectorMath {

    /** dot(query, center) * ||center||  — the KWS similarity score used on iOS. */
    fun kwsScore(query: FloatArray, center: FloatArray): Float {
        var dot = 0f
        var sumSq = 0f
        for (i in query.indices) {
            dot += query[i] * center[i]
            sumSq += center[i] * center[i]
        }
        return dot * sqrt(sumSq)
    }
}
