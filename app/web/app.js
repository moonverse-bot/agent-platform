/*
 * 页面逻辑:登录、会话列表、聊天(SSE 流式)、工单面板。
 * 刻意不引入任何框架和 CDN,一个文件跑完;渲染一律用 textContent,
 * 不用 innerHTML,避免模型输出里的尖括号变成 XSS。
 */
(function () {
  'use strict';

  var TOKEN_KEY = 'agent-platform-token';

  var state = {
    token: localStorage.getItem(TOKEN_KEY) || '',
    user: null,
    conversationId: null,
    conversations: [],
    sending: false,
  };

  var STATUS_LABEL = { open: '待处理', processing: '处理中', closed: '已关闭' };
  var PRIORITY_LABEL = { low: '低', normal: '普通', high: '高', urgent: '紧急' };

  var $ = function (id) { return document.getElementById(id); };
  var loginLayer = $('login-layer');
  var appEl = $('app');
  var messagesEl = $('messages');

  /* ---------------- 基础工具 ---------------- */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  async function api(path, options) {
    options = options || {};
    var headers = Object.assign({ 'Content-Type': 'application/json' }, options.headers || {});
    if (state.token && !options.anonymous) {
      headers.Authorization = 'Bearer ' + state.token;
    }
    var response = await fetch(path, Object.assign({}, options, { headers: headers }));

    if (response.status === 401 && !options.anonymous) {
      signOut();
      throw new Error('登录状态已失效,请重新登录');
    }
    if (response.status === 204) return null;

    var text = await response.text();
    var data = null;
    try { data = text ? JSON.parse(text) : null; } catch (error) { data = null; }

    if (!response.ok) {
      var detail = data && data.detail;
      throw new Error(typeof detail === 'string' ? detail : '请求失败(' + response.status + ')');
    }
    return data;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  /* ---------------- 极简 Markdown ---------------- */
  /* 只处理 **加粗**、`代码`、标题、列表和表格,够用且不会引入依赖 */

  function appendInline(fragment, text) {
    var pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
    var last = 0;
    var match;
    while ((match = pattern.exec(text)) !== null) {
      if (match.index > last) {
        fragment.appendChild(document.createTextNode(text.slice(last, match.index)));
      }
      var token = match[0];
      if (token.slice(0, 2) === '**') {
        fragment.appendChild(el('strong', null, token.slice(2, -2)));
      } else {
        fragment.appendChild(el('code', null, token.slice(1, -1)));
      }
      last = match.index + token.length;
    }
    if (last < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(last)));
    }
  }

  /* 表格:表头行 + 分隔行(| --- | --- |)+ 若干数据行。模型很爱输出表格,
     不认的话界面上就是一排竖线,所以这里单独处理。 */

  function isTableRow(line) {
    return /^\s*\|.*\|\s*$/.test(line);
  }

  function isTableStart(lines, index) {
    if (index + 1 >= lines.length) return false;
    return isTableRow(lines[index]) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[index + 1]);
  }

  function splitCells(line) {
    var cells = line.trim().split('|');
    if (cells.length && cells[0].trim() === '') cells.shift();
    if (cells.length && cells[cells.length - 1].trim() === '') cells.pop();
    return cells.map(function (cell) { return cell.trim(); });
  }

  function tableRow(line, tag) {
    var row = document.createElement('tr');
    splitCells(line).forEach(function (cell) {
      var node = document.createElement(tag);
      appendInline(node, cell);
      row.appendChild(node);
    });
    return row;
  }

  function buildTable(lines, index) {
    var table = el('table', 'md-table');
    table.appendChild(tableRow(lines[index], 'th'));
    var cursor = index + 2; // 跳过表头与分隔行
    while (cursor < lines.length && isTableRow(lines[cursor])) {
      table.appendChild(tableRow(lines[cursor], 'td'));
      cursor += 1;
    }
    return { table: table, next: cursor };
  }

  function renderRich(text) {
    var fragment = document.createDocumentFragment();
    var lines = text.split('\n');
    var index = 0;
    var needsBreak = false; // 上一行是普通文本时才需要补 <br>(开头和表格后都不补)

    while (index < lines.length) {
      if (isTableStart(lines, index)) {
        if (needsBreak) fragment.appendChild(document.createElement('br'));
        var built = buildTable(lines, index);
        fragment.appendChild(built.table);
        index = built.next;
        needsBreak = true;
        continue;
      }

      if (needsBreak) fragment.appendChild(document.createElement('br'));

      var line = lines[index];
      var heading = line.match(/^\s*#{1,6}\s+(.*)$/);
      var bullet = line.match(/^\s*[-*]\s+(.*)$/);
      var numbered = line.match(/^\s*(\d+)[.、]\s+(.*)$/);
      if (heading) {
        fragment.appendChild(el('div', 'md-h', heading[1]));
      } else if (bullet) {
        fragment.appendChild(document.createTextNode('• '));
        appendInline(fragment, bullet[1]);
      } else if (numbered) {
        fragment.appendChild(el('span', 'md-num', numbered[1] + '. '));
        appendInline(fragment, numbered[2]);
      } else {
        appendInline(fragment, line);
      }

      needsBreak = true;
      index += 1;
    }
    return fragment;
  }

  /* ---------------- 消息区渲染 ---------------- */

  function emptyHint() {
    var box = el('div', 'empty-hint');
    box.appendChild(el('h2', null, '用一句话提问,助手自己去查'));
    box.appendChild(el('p', null, '它不会瞎猜——需要查数据时会自己调用工具,你还能看到它调了什么。'));
    var list = el('ul', null);
    ['我有哪些工单?', '退款政策是什么?', '帮我建一个工单:登录报错 500'].forEach(function (text) {
      list.appendChild(el('li', null, text));
    });
    box.appendChild(list);
    return box;
  }

  function showEmptyHint() {
    messagesEl.replaceChildren(emptyHint());
  }

  function clearEmptyHint() {
    var hint = messagesEl.querySelector('.empty-hint');
    if (hint) hint.remove();
  }

  function thinkingIndicator() {
    var box = el('span', 'thinking');
    box.appendChild(el('i'));
    box.appendChild(el('i'));
    box.appendChild(el('i'));
    box.appendChild(document.createTextNode(' 正在思考'));
    return box;
  }

  function appendMessage(role, text) {
    var wrap = el('div', 'msg ' + role);
    var bubble = el('div', 'bubble');
    if (text) bubble.appendChild(renderRich(text));
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function toolCard(data) {
    var card = el('div', 'tool-card' + (data.ok ? '' : ' failed'));
    var line = el('div', 'tool-line');
    line.appendChild(el('span', 'tool-name', '调用工具:' + data.name));
    line.appendChild(el('span', null, data.ok ? '成功' : '失败'));
    line.appendChild(el('span', null, data.latency_ms + ' ms'));
    card.appendChild(line);

    var details = el('details');
    details.appendChild(el('summary', null, '看调用参数和返回结果'));
    var pre = el('pre');
    pre.textContent = '参数:' + JSON.stringify(data.arguments) + '\n\n返回:' + (data.result || '');
    details.appendChild(pre);
    card.appendChild(details);
    return card;
  }

  /* ---------------- 登录 ---------------- */

  function setLoginMessage(text) {
    $('login-message').textContent = text || '';
  }

  function showLogin() {
    appEl.classList.add('hidden');
    loginLayer.classList.remove('hidden');
  }

  function signOut() {
    state.token = '';
    state.user = null;
    state.conversationId = null;
    localStorage.removeItem(TOKEN_KEY);
    showLogin();
  }

  async function doAuth(register) {
    var email = $('login-email').value.trim();
    var password = $('login-password').value;
    setLoginMessage('');
    if (password.length < 8) {
      setLoginMessage('密码至少 8 位');
      return;
    }
    try {
      if (register) {
        await api('/api/auth/register', {
          method: 'POST',
          anonymous: true,
          body: JSON.stringify({ email: email, password: password, display_name: email.split('@')[0] }),
        });
      }
      var token = await api('/api/auth/login', {
        method: 'POST',
        anonymous: true,
        body: JSON.stringify({ email: email, password: password }),
      });
      state.token = token.access_token;
      localStorage.setItem(TOKEN_KEY, state.token);
      await enterApp();
    } catch (error) {
      setLoginMessage(error.message);
    }
  }

  async function enterApp() {
    state.user = await api('/api/auth/me');
    $('current-user').textContent = state.user.email;
    loginLayer.classList.add('hidden');
    appEl.classList.remove('hidden');
    state.conversationId = null;
    showEmptyHint();
    await Promise.all([loadConversations(), loadTickets()]);
  }

  /* ---------------- 会话 ---------------- */

  async function loadConversations() {
    state.conversations = await api('/api/conversations');
    renderConversations();
  }

  function renderConversations() {
    var list = $('conversation-list');
    list.replaceChildren();
    if (!state.conversations.length) {
      list.appendChild(el('li', 'muted', '还没有历史会话'));
      return;
    }
    state.conversations.forEach(function (conversation) {
      var item = el('li', 'conv-item' + (conversation.id === state.conversationId ? ' active' : ''));
      item.appendChild(el('span', 'conv-title', conversation.title));
      var remove = el('button', 'conv-del', '×');
      remove.title = '删除会话';
      remove.addEventListener('click', async function (event) {
        event.stopPropagation();
        // 删除不可恢复(消息与工具轨迹会一起删掉),所以先确认一次
        if (!window.confirm('删除这条会话?聊天记录会一并删掉,且不能恢复。')) return;
        await api('/api/conversations/' + conversation.id, { method: 'DELETE' });
        if (conversation.id === state.conversationId) startNewChat();
        await loadConversations();
      });
      item.appendChild(remove);
      item.addEventListener('click', function () { openConversation(conversation.id); });
      list.appendChild(item);
    });
  }

  function startNewChat() {
    state.conversationId = null;
    showEmptyHint();
    renderConversations();
  }

  async function openConversation(id) {
    state.conversationId = id;
    var history = await api('/api/conversations/' + id + '/messages');
    messagesEl.replaceChildren();
    if (!history.length) {
      showEmptyHint();
    } else {
      history.forEach(function (message) {
        appendMessage(message.role === 'user' ? 'user' : 'assistant', message.content);
      });
    }
    renderConversations();
  }

  /* ---------------- 工单 ---------------- */

  async function loadTickets() {
    var results = await Promise.all([api('/api/tickets/stats'), api('/api/tickets')]);
    renderStats(results[0]);
    renderTickets(results[1]);
  }

  function renderStats(stats) {
    var box = $('ticket-stats');
    box.replaceChildren();
    [['open', '待处理'], ['processing', '处理中'], ['closed', '已关闭'], ['total', '合计']].forEach(function (pair) {
      var cell = el('div', 'stat');
      cell.appendChild(el('b', null, String(stats[pair[0]] === undefined ? 0 : stats[pair[0]])));
      cell.appendChild(el('span', null, pair[1]));
      box.appendChild(cell);
    });
  }

  function renderTickets(tickets) {
    var list = $('ticket-list');
    list.replaceChildren();
    if (!tickets.length) {
      list.appendChild(el('li', 'muted', '还没有工单。可以在这里新建,也可以直接让助手帮你建。'));
      return;
    }
    tickets.forEach(function (ticket) {
      var card = el('li', 'ticket');
      var top = el('div', 'ticket-top');
      top.appendChild(el('div', 'ticket-title', '#' + ticket.id + ' ' + ticket.title));
      var badges = el('div');
      badges.appendChild(el('span', 'badge ' + ticket.status, STATUS_LABEL[ticket.status] || ticket.status));
      if (ticket.priority === 'high' || ticket.priority === 'urgent') {
        badges.appendChild(document.createTextNode(' '));
        badges.appendChild(el('span', 'badge ' + ticket.priority, PRIORITY_LABEL[ticket.priority]));
      }
      top.appendChild(badges);
      card.appendChild(top);

      if (ticket.description) card.appendChild(el('div', 'ticket-desc', ticket.description));

      if (ticket.status !== 'closed') {
        var actions = el('div', 'ticket-actions');
        if (ticket.status === 'open') {
          actions.appendChild(statusButton(ticket.id, 'processing', '标记处理中'));
        }
        actions.appendChild(statusButton(ticket.id, 'closed', '关闭'));
        card.appendChild(actions);
      }
      list.appendChild(card);
    });
  }

  function statusButton(ticketId, status, label) {
    var button = el('button', 'ghost small', label);
    button.addEventListener('click', async function () {
      await api('/api/tickets/' + ticketId, { method: 'PATCH', body: JSON.stringify({ status: status }) });
      await loadTickets();
    });
    return button;
  }

  /* ---------------- 对话 ---------------- */

  function parseEvent(block) {
    var type = 'message';
    var dataLines = [];
    block.split('\n').forEach(function (line) {
      if (line.indexOf('event:') === 0) type = line.slice(6).trim();
      else if (line.indexOf('data:') === 0) dataLines.push(line.slice(5).trim());
    });
    if (!dataLines.length) return null;
    try {
      return { type: type, data: JSON.parse(dataLines.join('\n')) };
    } catch (error) {
      return null;
    }
  }

  async function sendMessage(text) {
    if (state.sending) return;
    state.sending = true;
    $('send-btn').disabled = true;

    clearEmptyHint();
    appendMessage('user', text);
    var answer = null;

    function ensureBubble() {
      if (!answer) {
        var wrap = el('div', 'msg assistant');
        var bubble = el('div', 'bubble');
        var content = el('span');
        content.appendChild(thinkingIndicator());
        bubble.appendChild(content);
        wrap.appendChild(bubble);
        messagesEl.appendChild(wrap);
        answer = { wrap: wrap, content: content, started: false };
        scrollToBottom();
      }
      return answer;
    }

    try {
      var response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + state.token },
        body: JSON.stringify({ message: text, conversation_id: state.conversationId }),
      });
      if (response.status === 429) throw new Error('请求太频繁了,稍后再试');
      if (response.status === 401) { signOut(); return; }
      if (!response.ok || !response.body) throw new Error('对话接口返回 ' + response.status);

      var reader = response.body.getReader();
      var decoder = new TextDecoder('utf-8');
      var buffer = '';
      var finished = false;

      while (!finished) {
        var chunk = await reader.read();
        finished = chunk.done;
        buffer += decoder.decode(chunk.value || new Uint8Array(), { stream: !finished });

        var boundary;
        while ((boundary = buffer.indexOf('\n\n')) >= 0) {
          var event = parseEvent(buffer.slice(0, boundary));
          buffer = buffer.slice(boundary + 2);
          if (!event) continue;

          if (event.type === 'meta') {
            if (event.data.conversation_id) state.conversationId = event.data.conversation_id;
          } else if (event.type === 'token') {
            var target = ensureBubble();
            if (!target.started) { target.content.replaceChildren(); target.started = true; }
            target.content.appendChild(document.createTextNode(event.data.text || ''));
            scrollToBottom();
          } else if (event.type === 'reset') {
            // 模型调工具前的铺垫文字,服务端已经声明作废,这里直接撤掉这一条
            if (answer) { answer.wrap.remove(); answer = null; }
          } else if (event.type === 'tool') {
            messagesEl.appendChild(toolCard(event.data));
            scrollToBottom();
          } else if (event.type === 'error') {
            var wrap = el('div', 'msg assistant error');
            var bubble = el('div', 'bubble');
            bubble.appendChild(document.createTextNode(event.data.message || '出错了'));
            wrap.appendChild(bubble);
            messagesEl.appendChild(wrap);
            scrollToBottom();
          } else if (event.type === 'done') {
            var finalBubble = ensureBubble();
            finalBubble.started = true;
            finalBubble.content.replaceChildren();
            if (event.data.content) {
              finalBubble.content.appendChild(renderRich(event.data.content));
            } else {
              finalBubble.content.appendChild(document.createTextNode('（本次没有生成内容）'));
            }
            if (event.data.conversation_id) state.conversationId = event.data.conversation_id;
            scrollToBottom();
          }
        }
      }
    } catch (error) {
      var failWrap = el('div', 'msg assistant error');
      var failBubble = el('div', 'bubble');
      failBubble.appendChild(document.createTextNode('请求失败:' + error.message));
      failWrap.appendChild(failBubble);
      messagesEl.appendChild(failWrap);
      scrollToBottom();
    } finally {
      state.sending = false;
      $('send-btn').disabled = false;
      await Promise.all([loadConversations(), loadTickets()]);
    }
  }

  /* ---------------- 事件绑定 ---------------- */

  $('login-form').addEventListener('submit', function (event) {
    event.preventDefault();
    doAuth(false);
  });
  $('register-btn').addEventListener('click', function () { doAuth(true); });
  $('logout-btn').addEventListener('click', signOut);
  $('new-chat-btn').addEventListener('click', startNewChat);
  $('refresh-tickets-btn').addEventListener('click', loadTickets);

  $('ticket-form').addEventListener('submit', async function (event) {
    event.preventDefault();
    var title = $('ticket-title').value.trim();
    if (!title) return;
    await api('/api/tickets', {
      method: 'POST',
      body: JSON.stringify({
        title: title,
        description: $('ticket-desc').value.trim(),
        priority: $('ticket-priority').value,
      }),
    });
    $('ticket-title').value = '';
    $('ticket-desc').value = '';
    await loadTickets();
  });

  var input = $('chat-input');
  input.addEventListener('input', function () {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  });
  input.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      $('chat-form').requestSubmit();
    }
  });

  $('chat-form').addEventListener('submit', function (event) {
    event.preventDefault();
    var text = input.value.trim();
    if (!text) return;
    input.value = '';
    input.style.height = 'auto';
    sendMessage(text);
  });

  /* ---------------- 启动 ---------------- */

  (async function boot() {
    if (state.token) {
      try {
        await enterApp();
        return;
      } catch (error) {
        /* token 过期或后端没起,落回登录页 */
      }
    }
    showLogin();
  })();
})();
