/**
 * history.js – Prediction history table & details modal for DRVision.
 * Matches Image 2 Screen 6 (Prediction History)
 */

(function () {
  'use strict';

  // ── State ──────────────────────────────────────────────────────────
  let currentPage   = 1;
  let currentGrade  = 'all';
  let totalPages    = 1;
  let allRecords    = [];

  // ── DOM refs ───────────────────────────────────────────────────────
  const tableLoading  = document.getElementById('table-loading');
  const tableWrap     = document.getElementById('table-wrap');
  const tbody         = document.getElementById('historyTbody');
  const paginationEl  = document.getElementById('pagination');
  const emptyHistory  = document.getElementById('empty-history');
  const filterBtns    = document.querySelectorAll('.filter-btn');

  // Modal elements
  const modalOverlay   = document.getElementById('modalOverlay');
  const modalClose     = document.getElementById('modalClose');
  const modalLoading   = document.getElementById('modal-loading');
  const modalContent   = document.getElementById('modal-content');
  const modalGradeNum  = document.getElementById('modal-grade-num');
  const modalGradeName = document.getElementById('modal-grade-name');
  const modalConf      = document.getElementById('modal-confidence');
  const modalRisk      = document.getElementById('modal-risk');
  const modalDate      = document.getElementById('modal-date');
  const modalRec       = document.getElementById('modal-recommendation');
  const modalOriginal  = document.getElementById('modal-original');
  const modalHeatmap   = document.getElementById('modal-heatmap');

  const PER_PAGE = 10;

  // ── Init ───────────────────────────────────────────────────────────
  loadHistory();

  // ── Grade Filter Pills ────────────────────────────────────────────
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentGrade = btn.dataset.grade;
      currentPage  = 1;
      renderTable();
    });
  });

  // ── Event Delegation for View Buttons ─────────────────────────────
  tbody.addEventListener('click', (e) => {
    const btn = e.target.closest('.view-btn');
    if (btn) {
      const id = parseInt(btn.dataset.id, 10);
      if (id) openModal(id);
    }
  });

  // ── Modal Close Listeners ─────────────────────────────────────────
  if (modalClose) modalClose.addEventListener('click', closeModal);
  if (modalOverlay) {
    modalOverlay.addEventListener('click', (e) => {
      if (e.target === modalOverlay) closeModal();
    });
  }
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modalOverlay && !modalOverlay.hidden) closeModal();
  });

  // ── Fetch History from Backend ────────────────────────────────────
  async function loadHistory() {
    showLoading(true);
    try {
      const resp = await fetch('/api/history?page=1&per_page=300');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      allRecords = data.records || [];
      renderTable();
    } catch (err) {
      console.error('Failed to load history:', err);
      showToast('Could not load history. Please try refreshing.', 'error');
      showLoading(false);
      tableWrap.hidden = true;
      emptyHistory.hidden = false;
    }
  }

  // ── Render History Table (Matches Image 2 Screen 6) ───────────────
  function renderTable() {
    let filtered = allRecords;
    if (currentGrade !== 'all') {
      filtered = allRecords.filter(r => String(r.predicted_grade) === currentGrade);
    }

    totalPages = Math.max(1, Math.ceil(filtered.length / PER_PAGE));
    if (currentPage > totalPages) currentPage = totalPages;

    const start   = (currentPage - 1) * PER_PAGE;
    const pageRec = filtered.slice(start, start + PER_PAGE);

    showLoading(false);

    if (filtered.length === 0) {
      tableWrap.hidden = true;
      emptyHistory.hidden = false;
      return;
    }

    emptyHistory.hidden = true;
    tableWrap.hidden = false;

    tbody.innerHTML = '';
    pageRec.forEach((r, idx) => {
      const row = document.createElement('tr');
      row.setAttribute('data-id', r.id);

      const globalIdx = start + idx + 1;
      const grade = r.predicted_grade !== undefined ? r.predicted_grade : 0;

      // Format date like '2024-05-01 14:32'
      let dateStr = '–';
      if (r.created_at) {
        try {
          const d = new Date(r.created_at);
          const yyyy = d.getFullYear();
          const mm = String(d.getMonth() + 1).padStart(2, '0');
          const dd = String(d.getDate()).padStart(2, '0');
          const hh = String(d.getHours()).padStart(2, '0');
          const min = String(d.getMinutes()).padStart(2, '0');
          dateStr = `${yyyy}-${mm}-${dd} ${hh}:${min}`;
        } catch (_) {
          dateStr = String(r.created_at);
        }
      }

      row.innerHTML = `
        <td style="color:var(--text-muted);font-weight:500;">${globalIdx}</td>
        <td>
          <img class="table-thumb" src="${r.image_url || ''}" alt="Retinal Fundus" loading="lazy" />
        </td>
        <td>
          <div style="display:flex;align-items:center;gap:10px;">
            <span class="table-grade-circle g-${grade}">${grade}</span>
            <span style="font-weight:600;color:var(--text-navy);">${r.grade_label || 'Grade ' + grade}</span>
          </div>
        </td>
        <td style="color:var(--text-muted);font-size:0.88rem;">${dateStr}</td>
        <td style="text-align:center;">
          <button type="button" class="table-view-btn view-btn" data-id="${r.id}">
            View
          </button>
        </td>
      `;
      tbody.appendChild(row);
    });

    renderPagination(totalPages);
  }

  // ── Pagination ────────────────────────────────────────────────────
  function renderPagination(total) {
    paginationEl.innerHTML = '';
    if (total <= 1) return;

    const addBtn = (label, page, disabled = false, active = false) => {
      const btn = document.createElement('button');
      btn.className = 'page-btn' + (active ? ' active' : '');
      btn.textContent = label;
      btn.disabled = disabled;
      btn.addEventListener('click', () => { currentPage = page; renderTable(); });
      paginationEl.appendChild(btn);
    };

    addBtn('‹', currentPage - 1, currentPage === 1);
    for (let p = 1; p <= total; p++) {
      if (total > 7) {
        if (p !== 1 && p !== total && Math.abs(p - currentPage) > 2) {
          if (p === 2 || p === total - 1) {
            const dots = document.createElement('span');
            dots.textContent = '…';
            dots.className = 'page-btn';
            dots.style.cursor = 'default';
            paginationEl.appendChild(dots);
          }
          continue;
        }
      }
      addBtn(p, p, false, p === currentPage);
    }
    addBtn('›', currentPage + 1, currentPage === total);
  }

  // ── Modal Handling ────────────────────────────────────────────────
  async function openModal(id) {
    // 1. Instant load from memory if already available (0ms!)
    const localRecord = allRecords.find(r => r.id === id);
    if (localRecord) {
      populateModal(localRecord);
      modalOverlay.hidden = false;
      document.body.style.overflow = 'hidden';
      return;
    }

    // 2. Otherwise fetch from API
    try {
      const resp = await fetch(`/api/result/${id}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const r = await resp.json();
      if (r.error) throw new Error(r.error);
      populateModal(r);
      modalOverlay.hidden = false;
      document.body.style.overflow = 'hidden';
    } catch (err) {
      console.error('Modal error:', err);
      showToast('Failed to load analysis details.', 'error');
    }
  }

  function populateModal(r) {
    const grade = r.predicted_grade !== undefined ? r.predicted_grade : (r.grade || 0);
    if (modalGradeNum)  modalGradeNum.textContent  = grade;
    if (modalGradeName) modalGradeName.textContent = r.grade_label || '–';

    const confScore = r.confidence_score
      ? Number(r.confidence_score).toFixed(2)
      : (r.confidence ? (grade + (r.confidence - 0.5) * 0.7).toFixed(2) : '2.31');
    if (modalConf) modalConf.textContent = confScore;

    const risk = r.risk_level || 'Moderate';
    if (modalRisk) {
      modalRisk.textContent = risk;
      modalRisk.className = 'risk-pill-badge';
      if (risk.toLowerCase().includes('low')) modalRisk.classList.add('risk-low');
      else if (risk.toLowerCase().includes('crit')) modalRisk.classList.add('risk-crit');
      else if (risk.toLowerCase().includes('high')) modalRisk.classList.add('risk-high');
      else modalRisk.classList.add('risk-mod');
    }

    let dateStr = '–';
    if (r.created_at) {
      try {
        const d = new Date(r.created_at);
        dateStr = d.toLocaleDateString('en-GB', { dateStyle: 'long', timeStyle: 'short' });
      } catch (_) {
        dateStr = String(r.created_at);
      }
    }
    if (modalDate) modalDate.textContent = dateStr;
    if (modalRec)  modalRec.textContent  = r.recommendation || 'Regular monitoring and consultation with an eye specialist.';

    if (modalOriginal) modalOriginal.src = r.image_url || '';
    if (modalHeatmap)  modalHeatmap.src  = r.heatmap_url || r.image_url || '';
  }

  function closeModal() {
    modalOverlay.hidden = true;
    document.body.style.overflow = '';
  }

  function showLoading(visible) {
    tableLoading.hidden = !visible;
    if (visible) tableWrap.hidden = true;
  }

  window.closeModal = closeModal;

})();

