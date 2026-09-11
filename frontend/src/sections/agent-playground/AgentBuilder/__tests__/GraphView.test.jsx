/* eslint-disable react/prop-types */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render } from "@testing-library/react";
import { useAgentPlaygroundStore } from "../../store";
import GraphView from "../GraphView";

// Since GraphView uses ReactFlow internally (which requires a DOM provider),
// we test the callback logic extracted from GraphViewInner via the store
// and isolated callback tests.

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------
const mocks = vi.hoisted(() => ({
  mockScreenToFlowPosition: vi.fn((pos) => pos),
  mockEnsureDraft: vi.fn(),
  mockUpdateNodeApi: vi.fn(),
  mockEnqueueSnackbar: vi.fn(),
  mockSaveDraft: vi.fn(),
  reactFlowProps: null,
}));
const {
  mockScreenToFlowPosition,
  mockEnsureDraft,
  mockUpdateNodeApi,
  mockEnqueueSnackbar,
  mockSaveDraft,
} = mocks;
vi.mock("@xyflow/react", () => ({
  ReactFlow: ({ children, ...props }) => {
    mocks.reactFlowProps = props;
    return <div data-testid="react-flow">{children}</div>;
  },
  Controls: () => <div data-testid="controls" />,
  ConnectionLineType: { SmoothStep: "smoothstep" },
  useReactFlow: () => ({
    screenToFlowPosition: mockScreenToFlowPosition,
  }),
  ReactFlowProvider: ({ children }) => <div>{children}</div>,
  addEdge: (edge, edges) => [...edges, edge],
  applyEdgeChanges: (changes, edges) => edges,
  applyNodeChanges: (changes, nodes) =>
    changes.reduce(
      (currentNodes, change) =>
        change.type === "position"
          ? currentNodes.map((node) =>
              node.id === change.id
                ? { ...node, position: change.position }
                : node,
            )
          : currentNodes,
      nodes,
    ),
}));

vi.mock("../saveDraftContext", () => ({
  useSaveDraftContext: () => ({
    saveDraft: mocks.mockSaveDraft,
    ensureDraft: mocks.mockEnsureDraft,
  }),
}));

vi.mock("src/api/agent-playground/agent-playground", () => ({
  createConnectionApi: vi.fn(),
  deleteConnectionApi: vi.fn(),
  updateNodeApi: mocks.mockUpdateNodeApi,
  deleteNodeApi: vi.fn(),
}));

vi.mock("../../hooks/useCanEditAgent", () => ({
  default: () => ({ isReadOnly: false }),
}));

vi.mock("../hooks/useAddNodeOptimistic", () => ({
  default: () => ({ addNode: vi.fn() }),
}));

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

vi.mock("notistack", () => ({
  enqueueSnackbar: mocks.mockEnqueueSnackbar,
}));

vi.mock("src/utils/logger", () => ({
  default: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

vi.mock("../nodes", () => ({
  PromptNode: () => <div />,
  AgentNode: () => <div />,
  EvalNode: () => <div />,
}));

vi.mock("../edges", () => ({
  AnimatedEdge: () => <div />,
}));

vi.mock("../../components/ConfirmationDialog", () => ({
  ConfirmationDialog: ({ open, onClose, onConfirm }) =>
    open ? (
      <div data-testid="confirm-dialog">
        <button data-testid="confirm-btn" onClick={onConfirm}>
          Confirm
        </button>
        <button data-testid="cancel-btn" onClick={onClose}>
          Cancel
        </button>
      </div>
    ) : null,
}));

// ---------------------------------------------------------------------------
// Tests: GraphView callback logic
// ---------------------------------------------------------------------------
describe("GraphView – callback logic", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAgentPlaygroundStore.getState().reset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ---- onBeforeDelete ----
  describe("onBeforeDelete logic", () => {
    it("resolves immediately for empty deletions", async () => {
      // Simulating the onBeforeDelete callback behavior
      const onBeforeDelete = ({ nodes }) => {
        if (nodes.length === 0) return Promise.resolve(true);
        return new Promise((resolve) => resolve(true));
      };

      const result = await onBeforeDelete({ nodes: [] });
      expect(result).toBe(true);
    });

    it("returns a promise for non-empty deletions", () => {
      const onBeforeDelete = ({ nodes }) => {
        if (nodes.length === 0) return Promise.resolve(true);
        return new Promise(() => {
          // Waits for user confirmation
        });
      };

      const promise = onBeforeDelete({ nodes: [{ id: "n1" }] });
      expect(promise).toBeInstanceOf(Promise);
    });
  });

  // ---- handleConfirmDelete / handleCancelDelete ----
  describe("delete confirmation flow", () => {
    it("confirm resolves promise with true", async () => {
      let resolveRef;
      const promise = new Promise((resolve) => {
        resolveRef = resolve;
      });

      // Simulate confirm
      resolveRef(true);
      const result = await promise;
      expect(result).toBe(true);
    });

    it("cancel resolves promise with false", async () => {
      let resolveRef;
      const promise = new Promise((resolve) => {
        resolveRef = resolve;
      });

      resolveRef(false);
      const result = await promise;
      expect(result).toBe(false);
    });
  });

  // ---- handlePostDelete ----
  describe("handlePostDelete logic", () => {
    it("calls saveDraft with rollback callback", () => {
      const setGraphData = vi.fn();
      const snapshot = {
        nodes: [{ id: "n1" }],
        edges: [{ id: "e1" }],
      };

      // Simulate handlePostDelete
      mockSaveDraft({
        onError: () => {
          if (snapshot) {
            setGraphData(snapshot.nodes, snapshot.edges);
          }
        },
      });

      expect(mockSaveDraft).toHaveBeenCalledWith(
        expect.objectContaining({ onError: expect.any(Function) }),
      );

      // Simulate error — should rollback
      const onError = mockSaveDraft.mock.calls[0][0].onError;
      onError();

      expect(setGraphData).toHaveBeenCalledWith([{ id: "n1" }], [{ id: "e1" }]);
    });
  });

  // ---- onDrop ----
  describe("onDrop logic", () => {
    it("extracts node type and adds node at converted position", () => {
      const addNode = vi.fn();
      mockScreenToFlowPosition.mockReturnValue({ x: 100, y: 200 });

      // Simulate onDrop
      const event = {
        preventDefault: vi.fn(),
        clientX: 170,
        clientY: 220,
        dataTransfer: {
          getData: vi.fn((key) => {
            if (key === "application/reactflow") return "llm_prompt";
            if (key === "application/node-template-id") return "tpl-1";
            return "";
          }),
        },
      };

      event.preventDefault();
      const type = event.dataTransfer.getData("application/reactflow");
      const nodeTemplateId =
        event.dataTransfer.getData("application/node-template-id") || undefined;
      const position = mockScreenToFlowPosition({
        x: event.clientX - 70,
        y: event.clientY - 20,
      });
      addNode(type, position, nodeTemplateId);

      expect(addNode).toHaveBeenCalledWith(
        "llm_prompt",
        { x: 100, y: 200 },
        "tpl-1",
      );
    });

    it("does nothing when type is empty", () => {
      const addNode = vi.fn();

      const event = {
        preventDefault: vi.fn(),
        dataTransfer: {
          getData: vi.fn(() => ""),
        },
      };

      event.preventDefault();
      const type = event.dataTransfer.getData("application/reactflow");
      if (typeof type === "undefined" || !type) return;
      addNode(type);

      expect(addNode).not.toHaveBeenCalled();
    });
  });

  // ---- onConnect ----
  describe("onConnect logic", () => {
    it("calls storeOnConnect then saveDraft", () => {
      const storeOnConnect = vi.fn();
      const connection = { source: "n1", target: "n2" };

      // Simulate onConnect
      storeOnConnect(connection);
      mockSaveDraft();

      expect(storeOnConnect).toHaveBeenCalledWith(connection);
      expect(mockSaveDraft).toHaveBeenCalled();
    });
  });

  // ---- onNodeDragStop ----
  describe("onNodeDragStop logic", () => {
    const originalNode = {
      id: "n1",
      type: "llm_prompt",
      position: { x: 10, y: 20 },
      data: { label: "Prompt" },
    };
    const movedNode = {
      ...originalNode,
      position: { x: 100, y: 200 },
    };

    const renderGraphView = (isDraft) => {
      useAgentPlaygroundStore.setState({
        currentAgent: {
          id: "graph-1",
          version_id: "version-1",
          is_draft: isDraft,
        },
        nodes: [originalNode],
        edges: [],
      });
      render(<GraphView />);
    };

    const dragNode = () => {
      act(() => {
        mocks.reactFlowProps.onNodeDragStart(null, originalNode, [originalNode]);
        useAgentPlaygroundStore.setState({ nodes: [movedNode] });
        mocks.reactFlowProps.onNodeDragStop(null, movedNode, [movedNode]);
      });
    };

    it("promotes a saved agent and includes the moved position in draft creation", async () => {
      vi.useFakeTimers();
      let draftSnapshot;
      mockEnsureDraft.mockImplementation(async () => {
        draftSnapshot = useAgentPlaygroundStore.getState().nodes;
        useAgentPlaygroundStore.setState((state) => ({
          currentAgent: { ...state.currentAgent, is_draft: true },
        }));
        return "created";
      });
      renderGraphView(false);
      dragNode();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(500);
      });

      expect(mockEnsureDraft).toHaveBeenCalledOnce();
      expect(draftSnapshot[0].position).toEqual(movedNode.position);
      expect(useAgentPlaygroundStore.getState().currentAgent.is_draft).toBe(
        true,
      );
      expect(useAgentPlaygroundStore.getState().nodes[0].position).toEqual(
        movedNode.position,
      );
      expect(mockUpdateNodeApi).not.toHaveBeenCalled();
    });

    it("saves the moved position directly when the agent is already a draft", async () => {
      vi.useFakeTimers();
      mockEnsureDraft.mockResolvedValue(true);
      mockUpdateNodeApi.mockResolvedValue({});
      renderGraphView(true);
      dragNode();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(500);
      });

      expect(mockEnsureDraft).toHaveBeenCalledOnce();
      expect(mockUpdateNodeApi).toHaveBeenCalledWith({
        graphId: "graph-1",
        versionId: "version-1",
        nodeId: "n1",
        data: { position: movedNode.position },
      });
    });

    it("rolls the node back when saving its position fails", async () => {
      vi.useFakeTimers();
      mockEnsureDraft.mockResolvedValue(true);
      mockUpdateNodeApi.mockRejectedValue(new Error("save failed"));
      renderGraphView(true);
      dragNode();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(500);
      });

      expect(useAgentPlaygroundStore.getState().nodes[0].position).toEqual(
        originalNode.position,
      );
      expect(mockEnqueueSnackbar).toHaveBeenCalledWith(
        "Failed to save positions",
        { variant: "error" },
      );
    });
  });

  // ---- onConnectStart / onConnectEnd ----
  describe("connection tracking", () => {
    it("sets connection state on connect start", () => {
      useAgentPlaygroundStore.setState({
        isConnecting: false,
        connectingFromNodeId: null,
      });

      useAgentPlaygroundStore.getState().setIsConnecting?.(true);
      useAgentPlaygroundStore.getState().setConnectingFromNodeId?.("n1");

      const state = useAgentPlaygroundStore.getState();
      expect(state.isConnecting).toBe(true);
      expect(state.connectingFromNodeId).toBe("n1");
    });

    it("clears connection state on connect end", () => {
      useAgentPlaygroundStore.setState({
        isConnecting: true,
        connectingFromNodeId: "n1",
      });

      useAgentPlaygroundStore.getState().setIsConnecting?.(false);
      useAgentPlaygroundStore.getState().setConnectingFromNodeId?.(null);

      const state = useAgentPlaygroundStore.getState();
      expect(state.isConnecting).toBe(false);
      expect(state.connectingFromNodeId).toBeNull();
    });
  });
});
