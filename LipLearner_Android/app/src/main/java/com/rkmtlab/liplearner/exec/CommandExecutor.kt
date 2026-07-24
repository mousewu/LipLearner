package com.rkmtlab.liplearner.exec

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.core.content.edit

/**
 * Executes a recognized command. Android has no direct equivalent of iOS Shortcuts, so this offers
 * a configurable mapping and a broadcast hook for automation apps (Tasker / MacroDroid / Automate).
 *
 * Resolution order for a command string:
 *   1. An explicit user mapping in SharedPreferences ("cmd:<label>" -> a target string):
 *        - "app:<packageName>"  launches that app
 *        - "url:<uri>"          opens the URI (deep link / web)
 *        - "intent:<action>"    fires an activity Intent with that action
 *   2. Otherwise: broadcast `com.rkmtlab.liplearner.COMMAND` with extra `command`, so an automation
 *      app can trigger a routine (the closest analogue to run-shortcut).
 * A short toast always confirms what fired.
 */
class CommandExecutor(private val context: Context) {

    companion object {
        const val BROADCAST_ACTION = "com.rkmtlab.liplearner.COMMAND"
        private const val PREFS = "command_map"
    }

    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun setMapping(label: String, target: String) = prefs.edit { putString("cmd:$label", target) }
    fun getMapping(label: String): String? = prefs.getString("cmd:$label", null)

    fun execute(label: String) {
        val target = prefs.getString("cmd:$label", null)
        when {
            target?.startsWith("app:") == true -> launchApp(target.removePrefix("app:"), label)
            target?.startsWith("url:") == true -> openUri(target.removePrefix("url:"), label)
            target?.startsWith("intent:") == true -> fireAction(target.removePrefix("intent:"), label)
            else -> broadcast(label)
        }
    }

    private fun launchApp(pkg: String, label: String) {
        val intent = context.packageManager.getLaunchIntentForPackage(pkg)
        if (intent != null) {
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(intent)
            toast("▶ $label")
        } else {
            toast("App not found: $pkg")
        }
    }

    private fun openUri(uri: String, label: String) {
        try {
            context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(uri))
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            toast("▶ $label")
        } catch (e: Exception) {
            toast("Cannot open: $uri")
        }
    }

    private fun fireAction(action: String, label: String) {
        try {
            context.startActivity(Intent(action).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            toast("▶ $label")
        } catch (e: Exception) {
            toast("No app for: $action")
        }
    }

    private fun broadcast(label: String) {
        context.sendBroadcast(Intent(BROADCAST_ACTION).putExtra("command", label))
        toast("👄 $label")
    }

    private fun toast(msg: String) =
        Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
}
